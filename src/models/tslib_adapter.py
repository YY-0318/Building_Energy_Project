# src/models/tslib_adapter.py
import sys, os, torch, torch.nn as nn

# 动态挂载路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

class TSLib_Adapter(nn.Module):
    def __init__(self, model_name, args, input_dim):
        super().__init__()
        # 动态导入模型
        try:
            module = __import__(model_name)
            ModelNode = getattr(module, 'Model')
        except ImportError as e:
            # 如果还是报错，尝试从包路径导入 (针对部分复杂的库引用)
            import importlib
            module = __import__(model_name)
            ModelNode = getattr(module, 'Model')
        
        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        self.label_len = args.seq_len // 2  # TSLib 惯用的 Start Token 长度
        
        class Configs:
            def __init__(self):
                self.task_name = 'short_term_forecast'
                self.seq_len = args.seq_len
                self.pred_len = args.pred_len
                self.label_len = args.seq_len // 2 
                self.enc_in = input_dim
                self.dec_in = input_dim
                self.c_out = input_dim # 保持维度一致，稍后在 forward 切片
                self.d_model = args.d_model
                self.n_heads = args.n_heads
                self.e_layers = 2
                self.d_layers = 1
                self.d_ff = args.d_model * 4
                self.factor = 1
                self.dropout = 0.1
                self.activation = 'gelu'
                self.output_attention = False
                self.embed = 'timeF'
                self.freq = 'h'
                self.moving_avg = 25
                self.version = 'Fourier'
                self.mode_select = 'random'
                self.modes = 64
                self.top_k = 5
                self.num_kernels = 6
                self.seg_len = 12
                self.win_size = 2

        self.model = ModelNode(Configs())

    def forward(self, x):
        # x 形状: [Batch, Seq_len, Features]
        batch_size = x.shape[0]
        
        # 1. 准备 x_mark_enc (Encoder 时间特征)
        # 你的 batch_x 已经包含了时间特征，TSLib 通常分离开，
        # 这里为了不改 Trainer，给它全 0 的占位符（大多数模型在 Embed 层会处理）
        x_mark_enc = torch.zeros(batch_size, self.seq_len, 4).to(x.device)
        
        # 2. 准备 x_dec (Decoder 输入)
        # 由两部分组成: [已有序列的后一半 (Start Token), 待预测位置的占位符 (全0)]
        dec_inplace = torch.zeros(batch_size, self.pred_len, x.shape[-1]).to(x.device)
        x_dec = torch.cat([x[:, -self.label_len:, :], dec_inplace], dim=1)
        
        # 3. 准备 x_mark_dec (Decoder 时间特征)
        x_mark_dec = torch.zeros(batch_size, self.label_len + self.pred_len, 4).to(x.device)
        
        # 4. 调用 TSLib 模型 (适配标准的 4 输入签名)
        # 对于 iTransformer 这种只需要单输入的模型，TSLib 内部也会兼容处理
        output = self.model(x, x_mark_enc, x_dec, x_mark_dec)
        
        if isinstance(output, tuple):
            output = output[0]
            
        # 5. 降维处理：只取第一列能耗预测值 [Batch, Pred_Len]
        if output.dim() == 3:
            return output[:, -self.pred_len:, 0], None, None
        return output[:, -self.pred_len:], None, None