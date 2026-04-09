import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.trainer_short import EarlyStopping

# =======================================================
# 【核心创新 1】：时频联合损失函数 (Time-Freq Joint Loss)
# 彻底解决“波形形状对齐，但存在轻微时序平移”时的双倍惩罚问题
# =======================================================
class TimeFreqLoss(nn.Module):
    def __init__(self, alpha=0.95):
        super(TimeFreqLoss, self).__init__()
        self.alpha = alpha
        # 抛弃僵硬的 L1，换成带二次平滑的 SmoothL1Loss
        self.loss_fn = nn.SmoothL1Loss(beta=0.1)
        
    def forward(self, pred, true):
        # 1. 时域绝对误差 (Time Domain L1)
        loss_time = self.loss_fn(pred, true)
        
        # 2. 频域形状误差 (Frequency Domain L1)
        # 使用 rfft 提取实数序列的频域特征
        pred_fft = torch.fft.rfft(pred, dim=1, norm="forward")
        true_fft = torch.fft.rfft(true, dim=1, norm="forward")
        
        # 只取幅值 (Amplitude)，忽略相位 (Phase)
        # 这意味着：即使波形有 1-2 小时的平移偏移，只要振幅和周期对了，频域误差就是 0！
        pred_amp = torch.abs(pred_fft)
        true_amp = torch.abs(true_fft)
        loss_freq = self.loss_fn(pred_amp, true_amp)
        
        # 联合输出
        return self.alpha * loss_time + (1 - self.alpha) * loss_freq

class TrainerLong:
    def __init__(self, model, train_loader, val_loader, test_loader, scaler, test_dates, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.criterion = TimeFreqLoss(alpha=0.95) # 强烈约束时序相位
        
        # =======================================================
        # 【核心修复】: 参数隔离 (Parameter Isolation)
        # 绕过 PyTorch 的 _foreach_add 复数 Bug，同时符合数学严谨性
        # =======================================================
        real_params = []
        complex_params = []
        for p in self.model.parameters():
            if p.is_complex():
                complex_params.append(p)
            else:
                real_params.append(p)
                
        # 仅对实数参数 (Linear层) 施加 L2 正则化防过拟合
        # 频域复数参数免除正则化，保持纯净的傅里叶变换
        # 简单暴力的 L2 正则化，强行压制过拟合！
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=args.lr, 
            weight_decay=1e-4 
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=args.epochs,
            eta_min=1e-6
        )
    def calc_metrics(self, preds, trues):
        preds_flat = preds.flatten()
        trues_flat = trues.flatten()
        mae = mean_absolute_error(trues_flat, preds_flat)
        mse = mean_squared_error(trues_flat, preds_flat)
        r2 = r2_score(trues_flat, preds_flat)
        mape = np.mean(np.abs((trues_flat - preds_flat) / (trues_flat + 1e-5))) * 100
        return mae, mse, mape, r2

    def train(self):
        model_save_path = f"{self.args.out_dir}/models/best_model_{self.args.model}_long.pth"
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, path=model_save_path)
        
        print(f"Start Long-Term Training ({self.args.pred_len//24}-Day Horizon)...")
        
        train_losses = []
        val_losses_history = []
        
        for epoch in range(self.args.epochs):
            self.model.train()
            batch_losses = []
            for batch_x, batch_y, _, _ in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                
                # 动态解包 (兼容返回单个值或元组的模型)
                model_output = self.model(batch_x)
                output = model_output[0] if isinstance(model_output, tuple) else model_output
                
                loss = self.criterion(output, batch_y)
                loss.backward()
                # [新增] 梯度裁剪：防止验证集 Loss 震荡和突然飙升
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                batch_losses.append(loss.item())
            
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y, _, _ in self.val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    model_output = self.model(batch_x)
                    output = model_output[0] if isinstance(model_output, tuple) else model_output
                    val_losses.append(self.criterion(output, batch_y).item())
            
            avg_train, avg_val = np.mean(batch_losses), np.mean(val_losses)
            
            train_losses.append(avg_train)
            val_losses_history.append(avg_val)
            
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f} | LR: {current_lr:.6f}")
            
            # 步进学习率 (每个 epoch 结束时调用)
            self.scheduler.step()
            
            early_stopping(avg_val, self.model)
            if early_stopping.early_stop: 
                print("Early stopping triggered!")
                break
                
        # [修复 Bug]：将绘图代码移出 for epoch 循环，只在训练彻底结束后画一次
        print("Training finished. Plotting loss curve...")
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss') 
        plt.plot(val_losses_history, label='Val Loss') 
        plt.title('Long-Term Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{self.args.out_dir}/figures/loss_curve_long.png")
        plt.close()
        print(f"[Success] Loss curve saved to {self.args.out_dir}/figures/loss_curve_long.png")
            

    def test(self):
        best_path = f"{self.args.out_dir}/models/best_model_{self.args.model}_long.pth"
        self.model.load_state_dict(torch.load(best_path))
        self.model.eval()
        
        preds_all, trues_all = [], []
        features_all = [] # 用于 t-SNE
        
        with torch.no_grad():
            for batch_x, batch_y, _, _ in self.test_loader:
                batch_x = batch_x.to(self.device)
                
                model_output = self.model(batch_x)
                feat = None
                
                if isinstance(model_output, tuple):
                    output = model_output[0]
                    if len(model_output) >= 3:
                        feat = model_output[2]
                else:
                    output = model_output
                    
                preds_all.append(output.cpu().numpy())
                trues_all.append(batch_y.numpy())
                
                if feat is not None:
                    features_all.append(feat.mean(dim=1).cpu().numpy() if feat.dim()==3 else feat.cpu().numpy())
        
        preds_scaled = np.concatenate(preds_all, axis=0)
        trues_scaled = np.concatenate(trues_all, axis=0)
        
        # 计算归一化空间指标
        mae_norm, mse_norm, mape_norm, r2_norm = self.calc_metrics(preds_scaled, trues_scaled)
        
        # 逆转换 (还原为原始 kWh)
        sm, ss = self.scaler.mean_[0], self.scaler.scale_[0]
        preds_real = np.expm1(preds_scaled * ss + sm)
        trues_real = np.expm1(trues_scaled * ss + sm)
        preds_real = np.maximum(preds_real, 0)
        
        # 计算真实空间指标
        mae, mse, mape, r2 = self.calc_metrics(preds_real, trues_real)
        
        print(f"\n===== Academic Metrics (Long-Term: {self.args.pred_len}h) =====")
        print(f"Normalized | MAE: {mae_norm:.4f} | MSE: {mse_norm:.4f} | R2: {r2_norm:.4f}")
        print("-" * 55)
        print(f"Real Scale | MAE: {mae:.4f} | MSE: {mse:.4f}")
        print(f"Real Scale | MAPE: {mape:.4f}% | R2: {r2:.4f}")
        
        # =======================================================
        # 提取 30 天预测，按“天”聚合 (Daily Aggregation)
        # =======================================================
        num_days = self.args.pred_len // 24
        
        avg_preds_hourly = np.mean(preds_real, axis=0) 
        avg_trues_hourly = np.mean(trues_real, axis=0) 
        
        daily_preds = avg_preds_hourly.reshape(num_days, 24).sum(axis=1)
        daily_trues = avg_trues_hourly.reshape(num_days, 24).sum(axis=1)
        
        df_daily = pd.DataFrame({
            'Day': np.arange(1, num_days + 1),
            'True_Daily_Sum': daily_trues,
            'Pred_Daily_Sum': daily_preds
        })
        df_daily['Diff_Ratio'] = np.abs(df_daily['True_Daily_Sum'] - df_daily['Pred_Daily_Sum']) / (df_daily['True_Daily_Sum'] + 1e-5)
        
        csv_save_path = f"{self.args.out_dir}/preds/long_term_daily_agg_{self.args.model}.csv"
        df_daily.to_csv(csv_save_path, index=False)
        print(f"\n[Success] {num_days}天每日聚合预测已保存至: {csv_save_path}")
        
        # 绘制按天的柱状对比图
        plt.figure(figsize=(14, 6))
        bar_width = 0.35
        index = df_daily['Day']
        
        plt.bar(index - bar_width/2, df_daily['True_Daily_Sum'], bar_width, label='True Daily Energy', color='#2ca02c', alpha=0.7)
        plt.bar(index + bar_width/2, df_daily['Pred_Daily_Sum'], bar_width, label='Predicted Daily Energy', color='#d62728', alpha=0.8)
        
        plt.title(f'{num_days}-Day Long-Term Daily Energy Consumption Forecast', fontsize=14)
        plt.xlabel('Day in Forecast Horizon', fontsize=12)
        plt.ylabel('Daily Total Energy (kWh)', fontsize=12)
        plt.xticks(index)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{self.args.out_dir}/figures/long_term_daily_agg_{self.args.model}.png")
        plt.close()

        # =======================================================
        # t-SNE 特征分布可视化
        # =======================================================
        if len(features_all) > 0 and features_all[0] is not None:
            try:
                print("Running t-SNE visualization for Long-Term...")
                features_scaled = np.concatenate(features_all, axis=0)
                labels_energy = trues_scaled.mean(axis=1)
                labels_cat = pd.qcut(labels_energy, q=3, labels=["Low Energy", "Medium", "High Energy"])

                tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
                embedded = tsne.fit_transform(features_scaled)

                plt.figure(figsize=(10, 8), dpi=300)
                sns.scatterplot(x=embedded[:, 0], y=embedded[:, 1], hue=labels_cat, palette='viridis', alpha=0.7, edgecolor='w', s=60)
                plt.title("t-SNE Visualization of Fused Features (Long-Term)", fontsize=14, pad=15)
                plt.xlabel("t-SNE dim 1", fontsize=12)
                plt.ylabel("t-SNE dim 2", fontsize=12)
                plt.legend(title="Energy Level")
                plt.tight_layout()
                plt.savefig(f"{self.args.out_dir}/figures/tsne_long_term_{self.args.model}.png", bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"t-SNE failed: {e}")