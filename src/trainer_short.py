import torch
import time
import os
import torch.nn as nn
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        # 确保保存目录存在
        dir_name = os.path.dirname(self.path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, scaler, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.use_weighted_loss = getattr(args, 'use_weighted_loss', True)
        # 【修正点】在这里定义路径，并传给 EarlyStopping
        self.checkpoint_path = f"{args.out_dir}/models/checkpoint_{args.model}.pth"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.base_criterion = nn.L1Loss(reduction='none') 
        
        # --- 改进点 1: 引入权重衰减 (L2 正则化) ---
        # 通过 args.weight_decay 控制，通常建议设为 1e-4
        weight_decay = getattr(args, 'weight_decay', 1e-4)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=args.lr, 
            weight_decay=weight_decay
        )
        
        # --- 改进点 2: 引入学习率调度器 ---
        # 当验证集损失 5 个 epoch 不下降时，降低学习率为一半
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
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
        train_losses = []
        val_losses = []
        epoch_times = []
        model_save_path = f"{self.args.out_dir}/models/best_model_{self.args.mode}.pth"
        

        early_stopping = EarlyStopping(
            patience=getattr(self.args, 'patience', 20), 
            verbose=True, 
            path=self.checkpoint_path  
        )
        
        print(f"Start Training {self.args.mode} model for {self.args.epochs} epochs...")
        for epoch in range(self.args.epochs):
            epoch_start_time = time.time()
            self.model.train()
            batch_losses = []
            inf_start_time = time.time()
            lw = getattr(self.args, 'loss_weight', 5.0)
            for batch_x, batch_y, _, batch_hours in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                batch_hours = batch_hours.to(self.device)
                self.optimizer.zero_grad()
                
                # --- 【核心修改：健壮性解包】 ---
                model_output = self.model(batch_x)
                if isinstance(model_output, tuple):
                    output = model_output[0] # 始终取第一个作为预测值
                else:
                    output = model_output
                # ------------------------------
                
                
                if self.use_weighted_loss:
                    raw_loss = self.base_criterion(output, batch_y)
                    weights = torch.ones_like(raw_loss)
                    # 针对特定小时加权
                    difficult_hours = [8, 9, 11]
                    for h in difficult_hours: weights[batch_hours == h] = lw * 0.8
                    critical_hours = [10, 23]
                    for h in critical_hours: weights[batch_hours == h] = lw

                    weighted_loss = (raw_loss * weights).mean()
                    
                else:
                    # 标准平均损失 (消融实验用)
                    weighted_loss = self.base_criterion(output, batch_y).mean()
                
                weighted_loss.backward()
                self.optimizer.step()
                batch_losses.append(weighted_loss.item())
            
            
            
            # 验证环节
            self.model.eval()
            batch_val_losses = []
            with torch.no_grad():
                for batch_x, batch_y, _, _ in self.val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    # --- 【核心修改：验证环节同步解包】 ---
                    val_output_data = self.model(batch_x)
                    if isinstance(val_output_data, tuple):
                        val_output = val_output_data[0]
                    else:
                        val_output = val_output_data
                    # ------------------------------------
                    
                    loss = torch.mean(self.base_criterion(val_output, batch_y))
                    batch_val_losses.append(loss.item())
            inf_end_time = time.time()
            epoch_end_time = time.time()

            # 计算效率指标 (响应导师要求 5)
            avg_epoch_time = epoch_end_time - epoch_start_time
            epoch_times.append(avg_epoch_time)
            # 推理延迟 (ms/样本)
            inf_latency = ((inf_end_time - inf_start_time) / len(self.val_loader.dataset)) * 1000

            avg_train_loss = np.mean(batch_losses)      
            avg_val_loss = np.mean(batch_val_losses)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            
            # --- 改进点 4: 更新学习率调度器 ---
            self.scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{self.args.epochs} | LR: {self.optimizer.param_groups[0]['lr']:.6f}" 
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"Time: {avg_epoch_time:.2f}s | Inf: {inf_latency:.4f}ms/sample")
            
            early_stopping(avg_val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
        # 训练结束后打印总效率分析
        print(f"\n>>> Efficiency Analysis <<<")
        print(f"Average Training Time per Epoch: {np.mean(epoch_times):.4f} seconds")
        
        # 绘图逻辑保持不变
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'Training and Validation Loss Curve ({self.args.mode})')
        plt.legend()
        plt.savefig(f"{self.args.out_dir}/figures/loss_curve_{self.args.mode}.png")
        plt.close()

    def test(self):
        checkpoint_path = f"{self.args.out_dir}/models/checkpoint_{self.args.model}.pth"
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()
        
        preds_all, trues_all, dates_list = [], [], []
        # 【新增】用于存储权重和特征
        weights_all = []
        features_all = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_dates, _ in self.test_loader:
                batch_x = batch_x.to(self.device)
                
                # --- 【核心修改：动态解包】 ---
                model_output = self.model(batch_x)
                
                # 初始化默认值
                weights = None
                feat = None
                
                if isinstance(model_output, tuple):
                    output = model_output[0]  # 第一个是预测值
                    # 根据元组长度提取额外信息（TQNet 返回 3 个，LSTM/PatchTST 可能返回 1-2 个）
                    if len(model_output) >= 2:
                        weights = model_output[1]
                    if len(model_output) >= 3:
                        feat = model_output[2]
                else:
                    output = model_output
                # ------------------------------
                
                preds_all.append(output.cpu().numpy())
                trues_all.append(batch_y.numpy())
                dates_list.append(np.array(batch_dates).T)
                
                # 如果模型输出了特征，则收集用于 t-SNE
                if feat is not None:
                    # 取序列平均值或最后一个时刻的特征
                    features_all.append(feat.mean(dim=1).cpu().numpy() if feat.dim()==3 else feat.cpu().numpy())
                
                # 如果输出了权重，则收集用于 Attention Map
                if weights is not None:
                    weights_all.append(weights.cpu().numpy())
        
        preds_scaled = np.concatenate(preds_all, axis=0)
        trues_scaled = np.concatenate(trues_all, axis=0)
        dates_arr = np.concatenate(dates_list, axis=0)
        
        # 1. 计算归一化空间指标 (Normalized Metrics)
        mae_norm, mse_norm, mape_norm, r2_norm = self.calc_metrics(preds_scaled, trues_scaled)
        
        # 2. 逆转换到原始物理空间 (Real Scale)
        scale_mean = self.scaler.mean_[0]
        scale_std = self.scaler.scale_[0]
        preds_real = np.expm1(preds_scaled * scale_std + scale_mean)
        trues_real = np.expm1(trues_scaled * scale_std + scale_mean)
        preds_real = np.maximum(preds_real, 0)
        
        
        # 3. 计算真实空间指标 (Real Metrics)
        mae, mse, mape, r2 = self.calc_metrics(preds_real, trues_real)
        
        print(f"\n===== Academic Metrics (Full Window: {self.args.pred_len}h) =====")
        print(f"Normalized | MAE: {mae_norm:.4f} | MSE: {mse_norm:.4f} | R2: {r2_norm:.4f}")
        print("-" * 55)
        print(f"Real Scale | MAE: {mae:.4f} | MSE: {mse:.4f}")
        print(f"Real Scale | MAPE: {mape:.4f}% | R2: {r2:.4f}")
        
        # 4. 【核心改进：提取日前无重叠序列并计算“平均日负荷 (Diurnal Average)”】
        pred_len = self.args.pred_len  # 预测步长，如 24
        
        # (1) 先用跳跃采样获取真实的、无重叠的全量测试序列
        if len(preds_real.shape) == 3:
            continuous_preds = preds_real[::pred_len, :, -1].flatten() 
            continuous_trues = trues_real[::pred_len, :, -1].flatten()
        else:
            continuous_preds = preds_real[::pred_len, :].flatten()
            continuous_trues = trues_real[::pred_len, :].flatten()
            
        continuous_dates = dates_arr[::pred_len, :].flatten()
        
        min_len = min(len(continuous_dates), len(continuous_preds))
        continuous_dates = continuous_dates[:min_len]
        continuous_trues = continuous_trues[:min_len]
        continuous_preds = continuous_preds[:min_len]
        
        # (2) 放入 DataFrame
        import pandas as pd
        df_full = pd.DataFrame({
            'Date': continuous_dates,
            'True': continuous_trues,
            'Pred': continuous_preds
        })
        
        # (3) 提取小时信息 (0-23)，并按小时分组求平均
        df_full['Date'] = pd.to_datetime(df_full['Date'])
        df_full['Hour'] = df_full['Date'].dt.hour
        
        df_avg = df_full.groupby('Hour')[['True', 'Pred']].mean().reset_index()
        
        # 为了输出好看，把 Hour (如 8) 变成 '08:00' 格式
        df_avg['Time'] = df_avg['Hour'].astype(str).str.zfill(2) + ':00'
        
        # 计算平均曲线的 Diff_Ratio
        df_avg['Diff_Ratio'] = np.abs(df_avg['True'] - df_avg['Pred']) / (df_avg['True'] + 1e-5)
        
        # 整理最终输出列
        df_final_avg = df_avg[['Time', 'True', 'Pred', 'Diff_Ratio']]
        
        # 5. 保存到 CSV，文件名加上 seq_len 区分 24h 和 168h
        save_path = f"{self.args.out_dir}/preds/average_diurnal_{self.args.model}_seq{self.args.seq_len}.csv"
        df_final_avg.to_csv(save_path, index=False)
        print(f"\n【平均日负荷曲线】已聚合完成并保存至: {save_path}")
        print(f"输出维度: {df_final_avg.shape} (应为 24 行 4 列)")
        
        # 可选：直接在这里画图预览
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(df_final_avg['Time'], df_final_avg['True'], label='Average True', color='black', linewidth=2)
        plt.plot(df_final_avg['Time'], df_final_avg['Pred'], label=f'Average Pred (seq={self.args.seq_len})', color='red', linestyle='--', marker='o')
        plt.fill_between(df_final_avg['Time'], df_final_avg['True']*0.9, df_final_avg['True']*1.1, color='gray', alpha=0.2)
        plt.xticks(rotation=45)
        plt.title(f'Average Diurnal Load Profile (Seq_len={self.args.seq_len})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.args.out_dir}/figures/average_diurnal_{self.args.model}_seq{self.args.seq_len}.png")
        plt.close()
        
        # =======================================================
        # ① Attention Map 可视化 (取测试集第一个样本)
        # =======================================================
        if len(weights_all) > 0 and weights_all[0] is not None:
            plt.figure(figsize=(10, 8), dpi=300)
            # 取第一个 batch 的第一个样本的注意力图 [pred_len, seq_len]
            sample_weights = weights_all[0][0] 
            sns.heatmap(sample_weights, cmap='rocket_r', cbar_kws={'label': 'Attention Weight'})
            plt.title("Attention Map: Prediction Horizon vs. History Window", fontsize=15, pad=20)
            plt.xlabel("History Time Steps (168h)", fontsize=12)
            plt.ylabel("Forecast Time Steps (24h)", fontsize=12)
            # 设置坐标轴刻度，每 24 小时显示一个刻度，方便审稿人查看周期性
            plt.xticks(np.arange(0, 168+1, 24), labels=[f"-{i}h" for i in np.arange(168, -1, -24)])
            
            plt.tight_layout()
            plt.savefig(f"{self.args.out_dir}/figures/attention_map.png", bbox_inches='tight')
            plt.close()

        # =======================================================
        # ② t-SNE 特征分布可视化
        # =======================================================
        if len(features_all) > 0 and features_all[0] is not None:
            try:
                print("Running t-SNE visualization...")
                features_scaled = np.concatenate(features_all, axis=0)
                # 为了演示可分性，我们可以用真实能耗的高低作为标签 (分 3 类)
                labels_energy = trues_scaled.mean(axis=1)
                labels_cat = pd.qcut(labels_energy, q=3, labels=["Low Energy", "Medium", "High Energy"])

                tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
                embedded = tsne.fit_transform(features_scaled)

                plt.figure(figsize=(10, 8), dpi=300)
                scatter = sns.scatterplot(x=embedded[:, 0], y=embedded[:, 1], hue=labels_cat, palette='coolwarm', alpha=0.7, edgecolor='w', s=60)
                plt.title("t-SNE Visualization of Fused Features", fontsize=14, pad=15)
                plt.xlabel("t-SNE dimension 1", fontsize=12)
                plt.ylabel("t-SNE dimension 2", fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.legend(title="Energy Level", bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plt.tight_layout()
                plt.savefig(f"{self.args.out_dir}/figures/tsne_feature_distribution.png", bbox_inches='tight')
                plt.close()
                print(f"Visualizations saved to {self.args.out_dir}/figures/")
            except Exception as e:
                print(f"Warning: t-SNE visualization failed with error: {e}")
        else:
            print(f"Skip t-SNE: Model {self.args.model} did not provide feature vectors.")
        