import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体为顶刊常用的 Times New Roman  serif 字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False 

# 核心数据 (从你的文本中提取)
params_data = {
    'L': {'vals': [12, 24, 72, 168], 'mae': [300.76, 195.37, 220.6, 227.42], 'mape': [14.15, 6.78, 7.49, 8.11]},
    'K': {'vals': [3, 5, 7], 'mae': [195.37, 204.15, 208.61], 'mape': [6.78, 7.04,7.17]},
    'H': {'vals': [2, 4, 8], 'mae': [212.96, 195.37, 206.07], 'mape': [7.31, 6.78, 7.04]},
    'd_{model}': {'vals': [16, 32, 64, 128], 'mae': [204.80, 195.37, 204.56, 199.92], 'mape': [7.11, 6.78, 6.98, 6.91]},
    'tau': {'vals': [0.25, 0.5, 1.0, 2.0], 'mae': [204.46, 195.37, 206.53, 201.30], 'mape': [7.03, 6.78, 7.06, 6.97]}
}

# 顶刊专业配色
color_mae = '#1A5276' # 深蓝 (沉稳，绝对误差)
color_mape = '#E74C3C' # 橙红 (醒目，相对误差)
color_opt = '#229954' # 深绿 (最优解)

titles = ['(a) Window Length ($L$)', '(b) Kernel Size ($K$)', 
          '(c) Attention Heads ($H$)', '(d) Dimension ($d_{model}$)', 
          '(e) Temperature ($\tau$)']

fig, axes = plt.subplots(1, 5, figsize=(22, 5)) # 增加宽度以防拥挤
fig.subplots_adjust(wspace=0.35, top=0.85) # 调整子图和顶部的间距

for i, (key, data) in enumerate(params_data.items()):
    ax1 = axes[i]
    x_labels = [str(val) for val in data['vals']]
    x_pos = np.arange(len(x_labels))
    
    # --- 绘制 MAE (左轴，折线散点图) ---
    # 彻底弃用柱状图，改用更高级的带标记折线
    ax1.plot(x_pos, data['mae'], color=color_mae, linestyle='-', linewidth=2.5, 
             marker='o', markersize=9, markerfacecolor='white', markeredgewidth=2,
             label='MAE (left)')
    ax1.set_xlabel('Parameter Value', fontsize=13)
    
    # 仅在第一个子图显示左轴标题
    if i == 0:
        ax1.set_ylabel('MAE', fontsize=13, color=color_mae, fontweight='bold')
    
    ax1.tick_params(axis='y', labelcolor=color_mae, labelsize=11)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=11)
    ax1.set_title(titles[i], fontsize=15, pad=15)
    
    # 轻量级网格线 (仅纵轴)
    ax1.grid(True, axis='y', linestyle='--', color='gray', alpha=0.3)

    # --- 绘制 MAPE (右轴，分层折线散点图) ---
    ax2 = ax1.twinx()  
    ax2.plot(x_pos, data['mape'], color=color_mape, linestyle='--', linewidth=2, 
             marker='s', markersize=8, markerfacecolor=color_mape, markeredgecolor='white',
             label='MAPE (\%) (right)')
    
    # 仅在最后一个子图显示右轴标题
    if i == 4:
        ax2.set_ylabel('MAPE (\%)', fontsize=13, color=color_mape, fontweight='bold')
    
    ax2.tick_params(axis='y', labelcolor=color_mape, labelsize=11)

    # --- 终极优化：显式标注最优解 (The "Star" Mark) ---
    # 找到所有参数中的最佳 MAPE 对应的索引 (注意：你所有数据的最佳 MAPE 都是 6.78%)
    # 但我们以每个参数在该数据集中的最低值为准
    opt_idx_mape = np.argmin(data['mape'])
    
    # 在 MAPE 折线上标记绿星星
    ax2.plot(x_pos[opt_idx_mape], data['mape'][opt_idx_mape], 
             marker='*', markersize=20, markerfacecolor=color_opt, markeredgecolor='white', 
             markeredgewidth=2, linestyle='None', zorder=10)
    
    # 在 MAE 折线上也标记绿星星
    opt_idx_mae = np.argmin(data['mae'])
    ax1.plot(x_pos[opt_idx_mae], data['mae'][opt_idx_mae], 
             marker='D', markersize=10, markerfacecolor=color_opt, markeredgecolor='white', 
             markeredgewidth=2, linestyle='None', zorder=10)
    
    # 添加贯穿最优点的垂直虚线
    ax1.axvline(x_pos[opt_idx_mape], color=color_opt, linestyle=':', linewidth=1.5, alpha=0.7)

# --- 优雅地放置图例 (Top Center) ---
# 合并左右两轴的图例，并标记星星符号
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# 手动添加最优解的图例标记
opt_star = plt.Line2D([0], [0], marker='*', color='None', markerfacecolor=color_opt, 
                       markeredgecolor='white', markeredgewidth=2, markersize=18, linestyle='None')
opt_diamond = plt.Line2D([0], [0], marker='D', color='None', markerfacecolor=color_opt, 
                          markeredgecolor='white', markeredgewidth=2, markersize=10, linestyle='None')

fig.legend(handles1 + handles2 + [opt_star, opt_diamond], 
           labels1 + labels2 + ['Optimal MAPE', 'Optimal MAE'], 
           loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=4, fontsize=13, frameon=False)

# 保存高清图表
plt.savefig('hyperparameter_sensitivity_tiered.png', dpi=600, bbox_inches='tight')
plt.show()