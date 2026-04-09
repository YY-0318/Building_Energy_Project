import torch
import torch.nn as nn
from src.layers.revin import RevIN

class LSTM_Adapter(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, hidden_dim=128):
        super().__init__()
        self.revin = RevIN(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        self.revin._get_statistics(x)
        x = self.revin._normalize(x)
        
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1]) # 取最后一层的输出 [Batch, Pred_len]
        return out, None, h_n[-1]