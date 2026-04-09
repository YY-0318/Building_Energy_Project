import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 0. 全局样式设置 (顶刊双栏/单栏通用审美)
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12, 
    'font.family': 'serif',  # 使用学术界首选的衬线字体
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
})

def load_and_sort_data(filename):
    """读取 CSV 并按 00:00 - 23:00 重新排序"""
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Time'])
    df['Hour'] = df['Date'].dt.hour
    df = df.sort_values('Hour').reset_index(drop=True)
    return df

# ==========================================
# 1. 加载数据
# ==========================================
df_true = load_and_sort_data('date/TR-TQNet.csv') 
df_lstm = load_and_sort_data('date/LSTM.csv')
df_itransformer = load_and_sort_data('date/iTransformer.csv')
df_autoformer = load_and_sort_data('date/Autoformer.csv')
df_cross = load_and_sort_data('date/Crossformer.csv')
df_dlinear = load_and_sort_data('date/DLinear.csv')
df_fed          = load_and_sort_data('date/FEDformer.csv')
df_patchtst     = load_and_sort_data('date/PatchTST.csv')
df_vanilla      = load_and_sort_data('date/TQNet_Vanilla.csv') 

# 生成 X 轴标签格式：00:00, 01:00 ...
hours_labels = df_true['Hour'].astype(str).str.zfill(2) + ':00'

# ==========================================
# 2. 绘制 24 小时轨迹与包络图
# ==========================================
plt.figure(figsize=(12, 5))

# 绘制 ±10% 调度基准区间 (SBI) 灰色阴影
plt.fill_between(hours_labels, df_true['True'] * 0.97, df_true['True'] * 1.03, 
                 color='gray', alpha=0.25, label=r'$\pm 3\%$ SBI Envelope', linewidth=0)

# 绘制真实负荷 (黑色粗实线，放在较底层)
plt.plot(hours_labels, df_true['True'], color='black', linewidth=3, label='True Load', zorder=5)

# 绘制 TR-TQNet (红色实线带圆点，放在最顶层)
plt.plot(hours_labels, df_true['Pred'], color='#e63946', linewidth=2.5, 
         linestyle='-', marker='o', markersize=5, label='TR-TQNet (Ours)', zorder=6)

# 绘制 iTransformer (绿色虚线带方块)
plt.plot(hours_labels, df_itransformer['Pred'], color='#2a9d8f', linewidth=2, 
         linestyle='--', marker='s', markersize=4, label='iTransformer', zorder=4)

# 绘制 Autoformer (紫色点划线带三角)
plt.plot(hours_labels, df_autoformer['Pred'], color='#9c89b8', linewidth=2, 
         linestyle='-.', marker='^', markersize=4, label='Autoformer', zorder=3)

# 绘制 LSTM (蓝色点线带叉号)
plt.plot(hours_labels, df_lstm['Pred'], color='#457b9d', linewidth=2, 
         linestyle=':', marker='x', markersize=5, label='LSTM', zorder=2)

# 绘制 Crossformer
plt.plot(hours_labels, df_cross['Pred'], color='#f4a261', linewidth=2, 
         linestyle='--', marker='D', markersize=4, alpha=0.8, label='Crossformer', zorder=4)

plt.plot(hours_labels, df_dlinear['Pred'], color='#e9c46a', linewidth=2, 
         linestyle='-.', marker='v', markersize=5, alpha=0.8, label='DLinear', zorder=3)

plt.plot(hours_labels, df_fed['Pred'], color='#e07a5f', linewidth=2, 
         linestyle=':', marker='p', markersize=5, alpha=0.8, label='FEDformer', zorder=2)

plt.plot(hours_labels, df_patchtst['Pred'], color='#0077b6', linewidth=2, 
         linestyle='--', marker='*', markersize=6, alpha=0.8, label='PatchTST', zorder=4)

plt.plot(hours_labels, df_vanilla['Pred'], color='#8d99ae', linewidth=2, 
         linestyle='-.', marker='h', markersize=5, alpha=0.8, label='Vanilla TQNet', zorder=3)

# ==========================================
# 3. 细节修饰与保存
# ==========================================
plt.xticks(rotation=45)
plt.xlabel('Time of Day')
plt.ylabel('Load (kWh)')

# 将图例分两列排放，防止遮挡曲线上方的极值点
plt.legend(loc='upper left', ncol=2, framealpha=0.9) 
plt.title('24-hour Load Forecasting Trajectory and $\pm 3\%$ SBI Envelope', fontsize=14, pad=15)
plt.tight_layout()

# 保存为 300 DPI 的高清图片
plt.savefig('fig1_trajectory_all.png', dpi=300, bbox_inches='tight')
print("图 1 已成功生成：fig1_trajectory_updated.png")