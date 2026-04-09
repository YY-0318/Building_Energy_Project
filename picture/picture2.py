import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 0. 全局样式设置 (顶刊双栏/单栏通用审美)
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12, 
    'font.family': 'serif',  
    'axes.labelsize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

def load_and_sort_data(filename):
    """读取 CSV 并按 00:00 - 23:00 重新排序"""
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Time'])
    df['Hour'] = df['Date'].dt.hour
    df = df.sort_values('Hour').reset_index(drop=True)
    return df

# ==========================================
# 1. 加载数据 (提取正文中提到的典型模型)
# ==========================================
df_true     = load_and_sort_data('date/TR-TQNet.csv') 
df_cross    = load_and_sort_data('date/Crossformer.csv')
df_dlinear  = load_and_sort_data('date/DLinear.csv')
df_fed      = load_and_sort_data('date/FEDformer.csv')

# 生成 X 轴标签格式：00:00, 01:00 ...
hours_labels = df_true['Hour'].astype(str).str.zfill(2) + ':00'

# ==========================================
# 2. 截取早间爬坡段 (07:00 - 12:00)
# ==========================================
idx_start, idx_end = 7, 12  # 对应 07:00 到 12:00 (共6个数据点)

h_zoom = hours_labels.iloc[idx_start:idx_end+1]
true_zoom = df_true['True'].iloc[idx_start:idx_end+1]

plt.figure(figsize=(8, 5.5))

# 绘制 ±10% SBI 灰色阴影包络带
plt.fill_between(h_zoom, true_zoom * 0.97, true_zoom * 1.03, 
                 color='gray', alpha=0.2, label=r'$\pm 10\%$ SBI', linewidth=0)

# 绘制真实负荷 (黑色粗实线，底层)
plt.plot(h_zoom, true_zoom, color='black', linewidth=3.5, label='True Load', zorder=5)

# 绘制 TR-TQNet (红色实线，大圆点，顶层)
plt.plot(h_zoom, df_true['Pred'].iloc[idx_start:idx_end+1], 
         color='#e63946', linewidth=2.5, linestyle='-', marker='o', markersize=8, 
         label='TR-TQNet (Ours)', zorder=6)

# 绘制 Crossformer (深绿色点划线，方块) -> 展示它在极值点逼近红线
plt.plot(h_zoom, df_cross['Pred'].iloc[idx_start:idx_end+1], 
         color='#2a9d8f', linewidth=2, linestyle='-.', marker='s', markersize=6, 
         label='Crossformer', zorder=4)

# 绘制 DLinear (金色虚线，上三角) -> 展示它的相位滞后
plt.plot(h_zoom, df_dlinear['Pred'].iloc[idx_start:idx_end+1], 
         color='#e9c46a', linewidth=2, linestyle='--', marker='^', markersize=7, 
         label='DLinear', zorder=3)

# 绘制 FEDformer (紫色长虚线，菱形) -> 同样展示它对高频梯度的迟钝
plt.plot(h_zoom, df_fed['Pred'].iloc[idx_start:idx_end+1], 
         color='#9c89b8', linewidth=2, linestyle=':', marker='D', markersize=6, 
         label='FEDformer', zorder=2)

# ==========================================
# 3. 细节修饰 (标注 11:00 的最大需量结算点)
# ==========================================
max_demand_x = '11:00'
max_demand_y = df_true['True'].iloc[11]

# 画一根垂直的红色虚线提示 11:00
plt.axvline(x=max_demand_x, color='#c1121f', linestyle='--', alpha=0.5, zorder=1)

# # 添加带箭头的专业文本批注
# plt.annotate('Maximum Demand Point\n(Phase Locked by Ours)', 
#              xy=(max_demand_x, max_demand_y), 
#              xytext=('07:15', 5100), # 文本框位置
#              arrowprops=dict(facecolor='#343a40', shrink=0.05, width=1.5, headwidth=7),
#              fontsize=11, fontweight='bold', color='#343a40',
#              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.xlabel('Time of Day (Morning Ramping Period)')
plt.ylabel('Load (kWh)')

# 图例放在右下角，避免遮挡上方的爬坡曲线
plt.legend(loc='lower right', framealpha=0.9, fontsize=10)
plt.title('Transient Response during 07:00-11:00 Ramping Period', fontsize=14, pad=15)
plt.tight_layout()

# 保存为高清图片 (不在图内加 Title)
plt.savefig('fig2_ramping_updated.png', dpi=300, bbox_inches='tight')
print("图 2 已成功生成：fig2_ramping_updated.png")