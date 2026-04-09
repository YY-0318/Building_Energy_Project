import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 0. 全局样式设置 (符合顶刊审美)
# ==========================================
sns.set_theme(style="whitegrid")
# 使用衬线字体 (Serif) 更符合学术论文规范
plt.rcParams.update({
    'font.size': 12, 
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
})

# ==========================================
# 图 3: 帕累托前沿散点图 (Fig 3: Pareto Frontier)
# 数据直接提取自您的 LaTeX 表格
# ==========================================
models = ['LSTM', 'DLinear', 'Autoformer', 'Vanilla TQNet', 'FEDformer', 'PatchTST', 'iTransformer', 'Crossformer', 'TR-TQNet']
rmse = [595.61, 882.38, 484.99, 807.68, 425.51, 447.15, 387.79, 398.97, 315.21]
time = [0.5961, 0.5240, 4.2036, 0.5330, 19.7551, 2.3770, 1.4724, 8.2238, 1.1253]

plt.figure(figsize=(8, 6))

# 绘制散点
for i, model in enumerate(models):
    if model == 'TR-TQNet':
        plt.scatter(time[i], rmse[i], color='#e63946', s=250, marker='*', edgecolor='black', zorder=5)
        # 为 Ours 添加高亮文本
        plt.text(time[i]+0.8, rmse[i]-5, model, fontsize=12, fontweight='bold', color='#e63946')
    elif model == 'Crossformer':
        plt.scatter(time[i], rmse[i], color='#2a9d8f', s=100, marker='s', zorder=4)
        plt.text(time[i]-3.5, rmse[i]+10, model, fontsize=11, color='#2a9d8f', fontweight='bold')
    else:
        plt.scatter(time[i], rmse[i], color='#457b9d', s=80, alpha=0.7, zorder=3)
        plt.text(time[i]+0.8, rmse[i]-5, model, fontsize=10, alpha=0.8)

# 绘制辅助线 (突出 Ours 的优势区域)
plt.axhline(y=359.85, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=1.2255, color='gray', linestyle='--', alpha=0.5)
# 添加理想区域的高亮底色 (左下角)
plt.fill_betweenx([300, 359.85], -1, 1.2255, color='#e63946', alpha=0.08, label='Ideal Deployment Region')

plt.xlim(-1, max(time)+5)
plt.ylim(300, 900)
plt.xlabel('Computational Latency (s/epoch)')
plt.ylabel('RMSE (kWh)')
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.title('Pareto Frontier:Reliability(RMSE) vs. Computational Efficiency', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('fig3_pareto.png', dpi=300, bbox_inches='tight')
plt.close()
print("已生成: fig3_pareto.png")
print("全部图表生成完毕！")