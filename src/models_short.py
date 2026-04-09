import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    """
    可逆实例归一化 (Reversible Instance Normalization)
    用于消除时间序列的非平稳性 (Non-stationarity)
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
        x = x * self.stdev
        x = x + self.mean
        return x

class TQAttention(nn.Module):
    def __init__(self, pred_len, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.query_embedding = nn.Parameter(torch.randn(pred_len, d_model))
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, history_enc):
        batch_size = history_enc.shape[0]
        query = self.query_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        attn_out, _ = self.mha(query, history_enc, history_enc)
        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

class ShortTermTQNet(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, d_model=128, n_heads=4, kernel_size=3): 
        super().__init__()
        
        self.revin = RevIN(input_dim) 
        
        # [架构升级] 1. 特征投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # [架构升级] 2. 局部感知层 (CNN)
        # kernel_size=3, padding=1 保持序列长度不变
        # 这一层专门负责提取 "局部突变" (Local Gradients)
        padding_size = kernel_size // 2
        self.conv_local = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, padding=padding_size),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # 3. TQAttention (全局依赖)
        self.tq_block = TQAttention(pred_len, d_model)
        self.head_tq = nn.Linear(d_model, 1)
        
        # 4. Linear Trend (总体趋势)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [batch, seq_len, features]
        
        # RevIN Normalize
        self.revin._get_statistics(x) 
        x = self.revin._normalize(x)

        # Main Logic
        # 1. 投影到 d_model 维度
        x_emb = self.input_proj(x) + self.pos_enc
        
        # 2. [新增] 通过 CNN 提取局部特征
        # Conv1d 需要 [batch, channels, seq_len]
        x_conv = x_emb.permute(0, 2, 1) 
        x_conv = self.conv_local(x_conv)
        x_conv = x_conv.permute(0, 2, 1) # 变回 [batch, seq_len, d_model]
        
        # 残差连接：结合原始 embedding 和 局部特征
        x_enc = x_emb + x_conv
        
        # 3. TQAttention
        tq_out = self.tq_block(x_enc)
        residual = self.head_tq(tq_out).squeeze(-1)
        
        # 4. Trend
        x_target = x[:, :, 0] 
        trend = self.linear_trend(x_target)
        
        y_pred = trend + residual
        
        # RevIN Denormalize
        mean_target = self.revin.mean[:, 0, 0].unsqueeze(1)
        std_target = self.revin.stdev[:, 0, 0].unsqueeze(1)
        y_pred_scaled = y_pred * std_target + mean_target
        
        return y_pred_scaled
