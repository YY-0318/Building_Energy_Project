# src/models/__init__.py

# 注意：这里的 .tqnet_enhanced 必须对应你文件夹下的 tqnet_enhanced.py 文件
from .tqnet_enhanced import ShortTermTQNet 
from .dlinear import DLinear_Model
from .tqnet_vanilla import TQNet_Vanilla
from .lstm import LSTM_Adapter

def model_factory(model_name, config):
    if model_name == 'tqnet_enhanced':
        return ShortTermTQNet(**config)
    elif model_name == 'dlinear':
        return DLinear_Adapter(**config)
    elif model_name == 'tqnet_vanilla':
        return TQNet_Vanilla_Adapter(**config)
    elif model_name == 'lstm':
        return LSTM_Adapter(**config)
    else:
        raise ValueError(f"Unknown model: {model_name}")