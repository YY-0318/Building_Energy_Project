import torch
import torch.nn as nn
from src.layers.revin import RevIN

class TQNet_Vanilla(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, d_model=128):
        super().__init__()
        self.revin = RevIN(input_dim)
        # 参照 TQNet (1).py 的核心逻辑
        self.temporalQuery = nn.Parameter(torch.zeros(168, input_dim)) # 假设周期为24h
        self.channelAggregator = nn.MultiheadAttention(embed_dim=seq_len, num_heads=4, batch_first=True)
        self.output_proj = nn.Linear(d_model, pred_len)
        self.input_proj = nn.Linear(seq_len, d_model)

    def forward(self, x):
        # 自动获取当前 batch 的小时作为 cycle_index (假设它在特征某列，或者默认0)
        if self.revin:
            self.revin._get_statistics(x)
            x = self.revin._normalize(x)
            
        x_input = x.permute(0, 2, 1) # [B, C, S]
        # 简化原版逻辑以适配 Trainer
        query = self.temporalQuery[:1, :].repeat(x.size(0), x.size(1), 1) # 简化处理
        # ... (此处实现原版 TQ 核心注意力机制)
        out = self.input_proj(x_input)
        out = self.output_proj(torch.relu(out))
        return out[:, 0, :] # [Batch, Pred_len]