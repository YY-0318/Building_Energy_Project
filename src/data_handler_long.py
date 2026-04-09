import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import save_object # 关键点
from src.data_handler_short import FeatureSelector, TimeSeriesDataset

class TimeSeriesDatasetLong(TimeSeriesDataset):
    def __len__(self):
        # 允许窗口滑动到末尾
        length = len(self.data_x) - self.seq_len - self.pred_len
        return max(0, length)

def get_data(file_path, seq_len, pred_len, batch_size, save_dir, scaler_save_path=None):
    df = pd.read_csv(file_path)
    selector = FeatureSelector(save_dir=save_dir)
    df_processed = selector.process(df)
    
    # 原始时间提取
    raw_dates = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S').values
    raw_hours = pd.to_datetime(df['date']).dt.hour.values
    
    data = df_processed.values
    n = len(data)
    test_size = int(n * 0.15)
    val_size = int(n * 0.15)
    train_size = n - test_size - val_size
    
    train_set = TimeSeriesDatasetLong(data[:train_size], raw_dates[:train_size], raw_hours[:train_size], 
                                      seq_len, pred_len, is_train=True, use_log=True)
    
    # 使用 utils.py 保存 scaler
    if scaler_save_path:
        save_object(train_set.scaler, scaler_save_path)
    
    val_set = TimeSeriesDatasetLong(data[train_size-seq_len : train_size+val_size], 
                                    raw_dates[train_size-seq_len : train_size+val_size], 
                                    raw_hours[train_size-seq_len : train_size+val_size], 
                                    seq_len, pred_len, scaler=train_set.scaler, is_train=False, use_log=True)
    
    test_set = TimeSeriesDatasetLong(data[n-test_size-seq_len :], 
                                     raw_dates[n-test_size-seq_len :], 
                                     raw_hours[n-test_size-seq_len :], 
                                     seq_len, pred_len, scaler=train_set.scaler, is_train=False, use_log=True)

    return (DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(val_set, batch_size=batch_size, shuffle=False),
            DataLoader(test_set, batch_size=batch_size, shuffle=False),
            train_set.scaler, raw_dates)