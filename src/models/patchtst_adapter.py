# src/models/patchtst_adapter.py
import torch
import torch.nn as nn
import sys
import os

# =======================================================
# 路径自动修复逻辑：解决 PatchTST 内部 import layers 报错问题
# =======================================================
# 1. 获取当前适配器文件所在的物理目录 (src/models/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 检查该目录下是否存在 PatchTST 相关的文件夹或文件
# 确保当前目录在 sys.path 中，这样 PatchTST.py 才能找到它旁边的 layers 文件夹
if current_dir not in sys.path:
    sys.path.insert(0, current_dir) 

try:
    # 尝试导入。此时 Python 会在 current_dir 中寻找 PatchTST.py
    # 并且当 PatchTST.py 执行 'from layers...' 时，也能在 current_dir 下找到 layers/
    from PatchTST import Model as PatchTST_Model
except (ImportError, ModuleNotFoundError):
    # 备选：如果你的 PatchTST.py 在更深层的文件夹中，请修改此处的路径
    # 例如：from .PatchTST import Model as PatchTST_Model
    raise ImportError("无法加载 PatchTST 模型。请确保 PatchTST.py 和 layers 文件夹都在 src/models/ 目录下。")

# =======================================================

class PatchTST_Adapter(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, d_model=128, n_heads=4):
        super().__init__()
        # 创建一个临时的配置对象或直接传递参数
        class Config:
            pass
        
        configs = Config()
        configs.enc_in = input_dim      # 输入特征数
        configs.seq_len = seq_len      # 历史长度 (168)
        configs.pred_len = pred_len    # 预测长度 (24)
        configs.e_layers = 3           # Transformer 层数
        configs.n_heads = n_heads
        configs.d_model = d_model
        configs.d_ff = d_model * 4
        configs.dropout = 0.1
        configs.fc_dropout = 0.1
        configs.head_dropout = 0.1
        configs.patch_len = 16         # Patch 长度
        configs.stride = 8             # 步长
        configs.padding_patch = 'end'
        configs.revin = 1              # PatchTST 自带 RevIN
        configs.affine = 1
        configs.subtract_last = 0
        configs.decomposition = 0      # 是否使用趋势分解
        configs.kernel_size = 25
        configs.individual = 0         # 是否每个通道独立建模
        
        self.model = PatchTST_Model(configs)

    def forward(self, x):
        # PatchTST 默认输入: [Batch, Seq_len, Channel]
        # 它内部会处理维度和 Patching 逻辑
        output = self.model(x)
        if output.dim() == 3:
            # 将 [Batch, 24, 54] 降维为 [Batch, 24]
            output = output[:, :, 0]
            
        return output, None, None