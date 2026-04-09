import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import save_object

class FeatureSelector:
    def __init__(self, target_col='all_kwh', save_dir='./output/figures/'):
        self.target_col = target_col
        self.save_dir = save_dir
        self.leakage_cols = ['ac_kwh', 'shop_kwh', 'elevator_kwh', 'fire_kwh', 
                             'light_kwh', 'power_kwh', 'pump_kwh', 'special_kwh']
        self.internal_sensors = ['T_in_Upper', 'RH_in_Upper', 'T_in_Basement', 
                                 'RH_in_Basement', 'T_in_Avg', 'T_in_Std', 'RH_in_Avg', 
                                 'Delta_T_Simple', 'Delta_T_Lag_1h', 'Delta_T_Lag_3h']

    def process(self, df):
        # 0. 强制时间解析
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                df['hour'] = df['date'].dt.hour
                df['day_of_week'] = df['date'].dt.dayofweek
                df['month'] = df['date'].dt.month
                df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
                print("时间特征已重新校准 (Hour, DayOfWeek, Month)")
            except Exception as e:
                print(f"警告: 日期解析失败 {e}")

        # =======================================================
        # [核心改进] 智能分季节清洗
        # =======================================================
        if self.target_col in df.columns and 'month' in df.columns:
            # 定义夏季 (广州通常 5-10月 很热)
            is_summer = df['month'].isin([5, 6, 7, 8, 9, 10])
            
            # 规则：
            # 1. 非夏季 (11-4月): 阈值 6000 (处理 11月5日 7199 这种脏数据)
            # 2. 夏季 (5-10月): 阈值 12000 (保护真实高负荷)
            
            # 找出异常点 (同时满足非夏季且负荷过高)
            is_outlier = (~is_summer) & (df[self.target_col] > 6700)
            
            if is_outlier.any():
                count = is_outlier.sum()
                print(f"★ 智能清洗: 在非夏季检测到 {count} 个异常高值(>6700)，正在修复...")
                
                # 1. 先置空
                df.loc[is_outlier, self.target_col] = np.nan
                # 2. 尝试用上周填充
                fill_val = df[self.target_col].shift(168)
                df[self.target_col] = df[self.target_col].fillna(fill_val)
                # 3. 兜底插值
                df[self.target_col] = df[self.target_col].interpolate(method='linear').fillna(method='bfill')
                print("★ 清洗完成。夏季高负荷已保留。")
        # =======================================================

        # 1. 硬规则过滤
        drop_list = [c for c in self.leakage_cols if c in df.columns]
        drop_list += [c for c in self.internal_sensors if c in df.columns]
        drop_list += [c for c in df.columns if 'lag' in c]
        df_clean = df.drop(columns=drop_list, errors='ignore')
        
        # 2. 特征工程
        if 'schedule_open' in df_clean.columns:
            if 'wet_bulb_temperature' in df_clean.columns:
                df_clean['CDH'] = np.maximum(df_clean['wet_bulb_temperature'] - 18, 0) * df_clean['schedule_open']
            elif 'temperature' in df_clean.columns:
                df_clean['CDH'] = np.maximum(df_clean['temperature'] - 20, 0) * df_clean['schedule_open']
            
            if 'CDH' in df_clean.columns:
                df_clean['CDH_Rolling_3h'] = df_clean['CDH'].rolling(window=3, min_periods=1).mean()

            if 'hour' in df_clean.columns:
                df_clean['hours_from_open'] = df_clean['hour'] - 10
                df_clean['is_precooling'] = df_clean['hour'].apply(lambda x: 1 if 8 <= x < 10 else 0)
                df_clean['is_weekend_startup'] = df_clean['is_weekend'] * df_clean['is_precooling']
                
                df_clean['is_startup_window'] = df_clean['hour'].apply(lambda x: 1 if 8 <= x <= 11 else 0)
                df_clean['is_shutdown_window'] = df_clean['hour'].apply(lambda x: 1 if 22 <= x <= 23 else 0)

        if self.target_col in df_clean.columns:
            target = df_clean.pop(self.target_col)
            df_clean.insert(0, self.target_col, target)
            
            df_clean['lag24'] = df_clean[self.target_col].shift(24).fillna(method='bfill')
            df_clean['lag168'] = df_clean[self.target_col].shift(168).fillna(method='bfill')
            
            if 'is_weekend' in df_clean.columns:
                df_clean['weekend_lag168'] = df_clean['is_weekend'] * df_clean['lag168']
                df_clean['weekday_lag24'] = (1 - df_clean['is_weekend']) * df_clean['lag24']
            
            df_clean['ramp_24'] = df_clean['lag24'].diff().fillna(0)
            df_clean['mean_24_4h'] = df_clean['lag24'].rolling(window=4, min_periods=1).mean()
            
            if 'temperature' in df_clean.columns:
                df_clean['temp_diff_24'] = df_clean['temperature'] - df_clean['temperature'].shift(24).fillna(method='bfill')
                df_clean['temp_slope_3h'] = df_clean['temperature'].diff(3).fillna(0)

            # 最后的兜底截断 (防止梯度爆炸)
            limit = df_clean[self.target_col].quantile(0.99)
            df_clean[self.target_col] = df_clean[self.target_col].clip(upper=limit)

        df_clean = df_clean.fillna(method='ffill').fillna(0)
        
        df_numeric = df_clean.select_dtypes(include=[np.number])
        selected_features = df_numeric.columns.tolist()
        
        try:
            corr_matrix = df_numeric.corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
            plt.title("Feature Correlation Matrix")
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/correlation_heatmap.png")
            plt.close()
            print("[Info] Correlation heatmap saved.")
        except Exception as e:
            print(f"[Warning] Failed to plot heatmap: {e}")

        print(f"最终保留特征: {selected_features}")
        return df_clean[selected_features]

class TimeSeriesDataset(Dataset):
    def __init__(self, data, dates, hours, seq_len, pred_len, scaler=None, is_train=True, use_log=True):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dates = dates
        self.hours = hours
        self.use_log = use_log
        
        data_copy = data.copy()
        if self.use_log:
            data_copy[:, 0] = np.log1p(data_copy[:, 0])
        
        if is_train:
            self.scaler = StandardScaler()
            self.data_scaled = self.scaler.fit_transform(data_copy)
        else:
            self.scaler = scaler
            self.data_scaled = self.scaler.transform(data_copy)
            
        self.data_x = torch.FloatTensor(self.data_scaled)
        self.data_y = torch.FloatTensor(self.data_scaled[:, 0]) 

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        if r_end <= len(self.dates):
            seq_dates_np = self.dates[r_begin:r_end]
            seq_hours_np = self.hours[r_begin:r_end]
        else:
            seq_dates_np = self.dates[r_begin:len(self.dates)]
            seq_hours_np = self.hours[r_begin:len(self.hours)]
        
        return seq_x, seq_y, seq_dates_np.tolist(), torch.LongTensor(seq_hours_np)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len

def get_data(file_path, seq_len, pred_len, batch_size, save_dir, scaler_save_path=None):
    df = pd.read_csv(file_path)
    
    if 'date' in df.columns:
        raw_dates = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S').values
        raw_hours = pd.to_datetime(df['date']).dt.hour.values
    else:
        raw_dates = np.array([f"Step_{i}" for i in range(len(df))])
        if 'hour' in df.columns:
            raw_hours = df['hour'].values
        else:
            raw_hours = np.zeros(len(df))

    selector = FeatureSelector(save_dir=save_dir)
    df_processed = selector.process(df)
    data = df_processed.values
    
    n = len(data)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)
    
    train_data = data[:train_size]
    train_dates = raw_dates[:train_size]
    train_hours = raw_hours[:train_size]
    
    val_data = data[train_size:train_size+val_size]
    val_dates = raw_dates[train_size:train_size+val_size]
    val_hours = raw_hours[train_size:train_size+val_size]
    
    test_data = data[train_size+val_size:]
    test_dates = raw_dates[train_size+val_size:]
    test_hours = raw_hours[train_size+val_size:]
    
    train_set = TimeSeriesDataset(train_data, train_dates, train_hours, seq_len, pred_len, is_train=True, use_log=True)
    if scaler_save_path: save_object(train_set.scaler, scaler_save_path)
    
    val_set = TimeSeriesDataset(val_data, val_dates, val_hours, seq_len, pred_len, scaler=train_set.scaler, is_train=False, use_log=True)
    test_set = TimeSeriesDataset(test_data, test_dates, test_hours, seq_len, pred_len, scaler=train_set.scaler, is_train=False, use_log=True)
    
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(val_set, batch_size=batch_size, shuffle=False),
            DataLoader(test_set, batch_size=batch_size, shuffle=False),
            train_set.scaler)