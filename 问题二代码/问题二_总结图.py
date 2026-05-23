"""问题二总结图"""
import pandas as pd, numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for fn in ['Microsoft YaHei', 'SimSun', 'SimHei']:
    try:
        fm.findfont(fn, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [fn, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception: continue
plt.rcParams.update({'font.size':10,'axes.titlesize':14,'axes.labelsize':12,
                     'axes.titleweight':'bold','axes.labelweight':'bold'})

BASE = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE, '输出结果', '问题二')
CSV_DIR = os.path.join(OUT_DIR, '调度结果CSV')

df_s = pd.read_csv(os.path.join(CSV_DIR, 'SPT调度结果.csv'))
df_f = pd.read_csv(os.path.join(CSV_DIR, 'FIFO调度结果.csv'))
df_p = pd.read_csv(os.path.join(CSV_DIR, 'Priority调度结果.csv'))

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 1. Flow time distribution (counts, not density)
ax = axes[0, 0]
bins = np.linspace(0, 5000, 50)
for df_r, c, label in [(df_s, '#2196F3', 'SPT'), (df_f, '#FF9800', 'FIFO'), (df_p, '#4CAF50', 'Priority')]:
    ax.hist(df_r['流动时间(s)'], bins=bins, color=c, alpha=0.5,
            label=f'{label} (均值{df_r["流动时间(s)"].mean():.0f}s)')
ax.set_xlabel('流动时间 (秒)', fontsize=13, fontweight='bold')
ax.set_ylabel('箱数', fontsize=13, fontweight='bold')
ax.set_title('三种策略流动时间分布对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)

# 2. Per-aisle
ax = axes[0, 1]
x = np.arange(7); w = 0.25
for off, df_r, c, label in [(-w, df_s, '#2196F3', 'SPT'), (0, df_f, '#FF9800', 'FIFO'), (+w, df_p, '#4CAF50', 'Priority')]:
    means = [df_r[df_r['巷道号']==a+1]['流动时间(s)'].mean() for a in range(7)]
    ax.bar(x + off, means, w, color=c, edgecolor='white', label=label, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([f'巷道{a+1}' for a in range(7)])
ax.set_ylabel('平均流动时间 (秒)', fontsize=13, fontweight='bold')
ax.set_title('各巷道三策略对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)

# 3. Crane load
ax = axes[1, 0]
loads = df_s.groupby('巷道号').size()
colors_7 = plt.cm.tab10(np.linspace(0, 1, 7))
ax.bar(range(1, 8), [loads.get(a, 0) for a in range(1, 8)], color=colors_7, edgecolor='white')
ax.axhline(y=loads.mean(), color='gray', ls='--', label=f'均值 {loads.mean():.0f} 箱')
ax.set_xlabel('巷道号', fontsize=13, fontweight='bold')
ax.set_ylabel('处理箱数', fontsize=13, fontweight='bold')
ax.set_title('7台堆垛机负载分布', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)

# 4. Key metrics
ax = axes[1, 1]; ax.axis('off')
s_m, f_m, p_m = df_s['流动时间(s)'].mean(), df_f['流动时间(s)'].mean(), df_p['流动时间(s)'].mean()
worst = (df_p['流动时间(s)'] - df_s['流动时间(s)']).max()
info = f"""
问题二  调度优化  关键结论
================================

假设参数:
  堆垛机: 7台 (7巷道各1台)
  vx=0.2, vz=0.1 m/s (比2:1)
  匀速直线, 顺序运动, 空满载同速
  t0=6s/箱, 列宽400mm
  层高: 200/400/500mm (2:4:5)
  货架->巷道: (货架号-1)//2+1
  订单: 3738箱, 6.7天, 库位来自问题一

调度规则:
  SPT: 选运输时间最短的先做
  FIFO: 按到达顺序, 不重排
  Priority: 按消耗占比从高到低

结果:
  策略      平均流动时间    vs SPT
  SPT       {s_m:.0f}s       --
  Priority  {p_m:.0f}s       +{(p_m/s_m-1)*100:.0f}%
  FIFO      {f_m:.0f}s       +{(f_m/s_m-1)*100:.0f}%

  SPT中位数: {df_s['流动时间(s)'].median():.0f}s
  Priority最差单箱多等: {worst:.0f}s

  Priority不如SPT的原因:
  消耗占比高 != 运输时间短
  高消耗长T箱插队堵住短箱

  7机负载: {loads.min()}-{loads.max()} 箱/机
  利用率: 16-19%
  临界速度 vx=0.32 m/s
"""
ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=8.5,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#CCCCCC'))

plt.tight_layout(pad=2.5)
fig.savefig(os.path.join(OUT_DIR, '问题二总结.png'), dpi=150, bbox_inches='tight', pad_inches=0.3)
plt.close()
print("Done!")
