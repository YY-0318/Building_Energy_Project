import torch
import torch.nn as nn
from src.layers.revin import RevIN # 假设你提取了 RevIN

class DLinear_Model(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, use_revin=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(input_dim)
            
        # 参照你上传的 DLinear 逻辑
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        # 简单分解算子
        self.decompsition = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)

    def forward(self, x):
        # x: [Batch, Seq, Features]
        if self.use_revin:
            self.revin._get_statistics(x)
            x = self.revin._normalize(x)
            
        # 分解
        trend_init = self.decompsition(x.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal_init = x - trend_init
        
        # 映射并只取第一个通道 (all_kwh)
        seasonal_output = self.Linear_Seasonal(seasonal_init.permute(0, 2, 1))
        trend_output = self.Linear_Trend(trend_init.permute(0, 2, 1))
        
        x = seasonal_output + trend_output
        return x[:, 0, :] # [Batch, Pred_len]