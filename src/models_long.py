import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from src.models_short import RevIN, TQAttention

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class LongTermDualTQNet(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, d_model=128, patch_len=48, stride=24, top_k=15):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.top_k = top_k
        self.num_days_out = pred_len // patch_len 
        
        self.decomp = SeriesDecomp(kernel_size=25)
        self.feature_fusion = nn.Linear(input_dim, 1)
        # =========================================================
        # 1. 宏观趋势分支 (DTE: 阻尼趋势外推)
        # =========================================================
        self.trend_linear = nn.Linear(seq_len, pred_len)
        self.damp_predictor = nn.Sequential(
            nn.Linear(seq_len, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus() 
        )
        
        # =========================================================
        # 2. 每日自适应振幅 (DAAS)
        # =========================================================
        self.var_predictor = nn.Sequential(
            nn.Linear(seq_len, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.num_days_out), 
            nn.Sigmoid() 
        )
        
        # =========================================================
        # 3. 高容量微观分支 (彻底剔除复数参数 Bug)
        # =========================================================
        self.revin_target = RevIN(1) 
        self.patch_num = int((seq_len - patch_len) / stride + 1)
        self.patch_proj = nn.Linear(patch_len, d_model) 
        
        self.tq_block = TQAttention(self.patch_num, d_model, dropout=0.2) 
        freq_in = self.patch_num // 2 + 1
        
        # 【核心修复】：将原本的复数全连接层，拆解为两个纯实数全连接层！
        # 彻底绕过 PyTorch 的 _foreach_add 复数碰撞 Bug
        # self.freq_proj_real = nn.Linear(freq_in, freq_in)
        # self.freq_proj_imag = nn.Linear(freq_in, freq_in)
        
        # 在 __init__ 中修改频域线性层
        self.freq_proj = nn.Linear(freq_in * 2, freq_in * 2) # 合并处理
        
        self.head_seasonal = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(self.patch_num * d_model, pred_len)
        )

    def forward(self, x):
        # 1. 序列分解
        seasonal_init, trend_init = self.decomp(x)
        
        # 2. 阻尼趋势执行
        target_trend = trend_init[:, :, 0] 
        anchor = target_trend[:, -1:] 
        trend_aligned = target_trend - anchor 
        
        trend_delta_raw = self.trend_linear(trend_aligned) 
        
        tau = self.damp_predictor(trend_aligned) 
        t = torch.linspace(0, 1, self.pred_len, device=x.device).unsqueeze(0) 
        damp_mask = torch.exp(-tau * t) 
        
        trend_pred = anchor + (trend_delta_raw * damp_mask) 
        
        # 3. DAAS 每日独立振幅
        gamma = self.var_predictor(trend_aligned) * 2.0 
        gamma = gamma.unsqueeze(2).repeat(1, 1, self.patch_len) 
        gamma = gamma.view(x.shape[0], self.pred_len) 
        
        # 4. 高频波形恢复
        seasonal_target = seasonal_init[:, :, 0:1] 
        self.revin_target._get_statistics(seasonal_target)
        seasonal_norm = self.revin_target._normalize(seasonal_target)
        
        patches = []
        for i in range(self.patch_num):
            start = i * self.stride
            end = start + self.patch_len
            patch = seasonal_norm[:, start:end, 0] 
            patches.append(patch)
            
        patch_tensor = torch.stack(patches, dim=1) 
        x_emb = self.patch_proj(patch_tensor)      
        
        tq_out = self.tq_block(x_emb)
        
        xf = torch.fft.rfft(x_emb, dim=1)
        amplitudes = torch.abs(xf)
        k = min(self.top_k, xf.shape[1])
        _, topk_indices = torch.topk(amplitudes, k, dim=1)
        mask = torch.zeros_like(xf)
        mask.scatter_(1, topk_indices, 1.0)
        xf_filtered = xf * mask
        
        # 【核心修复执行】：对实部和虚部分别使用普通的实数 Linear 处理
#         xf_real = xf_filtered.real.permute(0, 2, 1)
#         xf_imag = xf_filtered.imag.permute(0, 2, 1)
        
#         xf_pred_real = self.freq_proj_real(xf_real).permute(0, 2, 1)
#         xf_pred_imag = self.freq_proj_imag(xf_imag).permute(0, 2, 1)
        
#         # 重新组合为复数信号送入 IFFT
#         xf_pred = torch.complex(xf_pred_real, xf_pred_imag)


        xf_real = xf_filtered.real.permute(0, 2, 1) # [Batch, d_model, freq_in]
        xf_imag = xf_filtered.imag.permute(0, 2, 1)
        
        # 核心修改：将实部和虚部在最后一个维度拼接起来，让它们产生信息交互
        xf_cat = torch.cat([xf_real, xf_imag], dim=-1) # 维度变成 [Batch, d_model, freq_in * 2]
        
        # 使用统一的 Linear 层进行处理
        xf_pred_cat = self.freq_proj(xf_cat)
        
        # 处理完后，再从中间一分为二，拆回实部和虚部
        freq_in = xf_real.shape[-1]
        xf_pred_real = xf_pred_cat[..., :freq_in].permute(0, 2, 1) 
        xf_pred_imag = xf_pred_cat[..., freq_in:].permute(0, 2, 1)
        
        # 重新组合为复数
        xf_pred = torch.complex(xf_pred_real, xf_pred_imag)
        
        freq_res_emb = torch.fft.irfft(xf_pred, n=self.patch_num, dim=1)
        
        fused_emb = tq_out + freq_res_emb
        
        seasonal_pred_norm = self.head_seasonal(fused_emb) 
        
        historical_std = self.revin_target.stdev[:, 0, 0].unsqueeze(1) 
        seasonal_pred = seasonal_pred_norm * historical_std * gamma
        
        y_pred_scaled = trend_pred + seasonal_pred
        
        return y_pred_scaled, None, fused_emb
