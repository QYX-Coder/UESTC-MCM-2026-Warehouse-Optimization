"""
四策略对比图：我们的方案 vs 随机 vs 逐层填满 vs 逐列填满
"""
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

plt.rcParams.update({'font.size':11,'axes.titlesize':14,'axes.labelsize':13,
                     'axes.titleweight':'bold','axes.labelweight':'bold',
                     'figure.titlesize':16,'figure.titleweight':'bold'})

BASE = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE, '输出结果', '问题一')
CSV_DIR = os.path.join(OUT_DIR, '分配结果CSV')

df_opt = pd.read_csv(os.path.join(CSV_DIR, '完整分配明细.csv'))
df1 = pd.read_csv(os.path.join(BASE, '清洗数据一', '清洗后_原材料库存数据.csv'))
N_S, N_L, N_C = 14, 50, 68
C_MAP = {'E1': '#2196F3', 'E3': '#FF9800', 'E4': '#4CAF50'}
L_MAP = {'E1': 'E1箱', 'E3': 'E3箱', 'E4': 'E4箱'}
COL_W = 400

LH = {}; cum = 0.0
for z in range(1, 51):
    if z <= 8: h = 0.2
    elif z <= 42: h = 0.4
    else: h = 0.5
    LH[z] = cum + h / 2; cum += h
VXE, VZE, VXL, VZL = 3.0, 0.75, 2.3, 0.58

# ============ Our time ============
def ct(r):
    Dx = r['列(x)'] * 0.4; Dz = LH[r['层(z)']]
    return Dx / VXE + Dz / VZE + Dx / VXL + Dz / VZL + 6.0

t_opt = np.array([ct(r) for _, r in df_opt.iterrows()])
tw_opt = np.average(t_opt, weights=df_opt['消耗占比'] / df_opt['消耗占比'].sum())

# ============ Simulate other strategies ============
def la(z):
    if z <= 8: return ['E1']
    elif z <= 42: return ['E1', 'E3']
    return ['E1', 'E3', 'E4']

def run_trial(seed, pos_order):
    mats = {'E1': [], 'E3': [], 'E4': []}
    for _, r in df1.iterrows():
        bt, q, c, mid = r['箱子类型'], int(r['库存数量/箱']), r['消耗占比'], r['原材料编号']
        mats[bt].append({'id': mid, 'cons': c, 'qty': q})
    rng = np.random.RandomState(seed)
    for bt in mats: rng.shuffle(mats[bt])
    rem = {}; idx = {'E1': 0, 'E3': 0, 'E4': 0}
    for bt in mats:
        for m in mats[bt]: rem[m['id']] = m['qty']

    alloc = []
    for col, layer in pos_order:
        for shelf in range(1, N_S + 1):
            best_bt, best_r, best_id = None, -1, None
            for bt in la(layer):
                while idx[bt] < len(mats[bt]) and rem[mats[bt][idx[bt]]['id']] <= 0: idx[bt] += 1
                if idx[bt] < len(mats[bt]) and mats[bt][idx[bt]]['cons'] > best_r:
                    best_r = mats[bt][idx[bt]]['cons']; best_bt = bt; best_id = mats[bt][idx[bt]]['id']
            if best_id is None: continue
            q = rem[best_id]
            if q >= 2:
                alloc.extend([{'列(x)': col, '层(z)': layer, '货架(y)': shelf, '消耗占比': best_r, '箱子类型': best_bt}] * 2)
                rem[best_id] -= 2
            else:
                alloc.append({'列(x)': col, '层(z)': layer, '货架(y)': shelf, '消耗占比': best_r, '箱子类型': best_bt})
                rem[best_id] -= 1
    df_sim = pd.DataFrame(alloc)
    t = np.array([ct(r) for _, r in df_sim.iterrows()])
    return np.average(t, weights=df_sim['消耗占比'] / df_sim['消耗占比'].sum()), df_sim

# Position orders
po_ours = [(c, l) for l in range(1, N_L + 1) for c in range(1, N_C + 1)]
po_ours.sort(key=lambda p: p[0] * 400 + LH[p[1]] * 1000)
po_layer = [(c, l) for l in range(1, N_L + 1) for c in range(1, N_C + 1)]
po_layer.sort(key=lambda p: (p[1], p[0]))
po_col = [(c, l) for l in range(1, N_L + 1) for c in range(1, N_C + 1)]
po_col.sort(key=lambda p: (p[0], p[1]))

print("Simulating strategies...")
tw_rand, df_rand = run_trial(0, po_ours)
tw_layer, df_layer = run_trial(0, po_layer)
tw_col, df_col = run_trial(0, po_col)

print(f"Our: {tw_opt:.2f}s, Random: {tw_rand:.2f}s, Layer: {tw_layer:.2f}s, Column: {tw_col:.2f}s")

# ============ FIGURE 1: 3D comparison ============
fig = plt.figure(figsize=(24, 20))
datasets = [
    (df_opt, f'优化方案 (距离排序+消耗排序)', tw_opt),
    (df_rand, f'随机打乱 (同填充)', tw_rand),
    (df_layer, f'逐层填满 (层优先)', tw_layer),
    (df_col, f'逐列填满 (列优先)', tw_col),
]
for i, (src, title, tm) in enumerate(datasets):
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    for bt, c in C_MAP.items():
        sub = src[src['箱子类型'] == bt]
        ax.scatter(sub['列(x)'], sub['货架(y)'], sub['层(z)'],
                   c=c, s=0.6, alpha=0.5, label=L_MAP[bt], rasterized=True)
    for z in [8.5, 42.5]:
        ax.plot([1, N_C], [N_S / 2] * 2, [z, z], 'gray', lw=1, ls=':', alpha=0.5)
    ax.set_xlabel('列 x', fontsize=13, fontweight='bold')
    ax.set_ylabel('货架 y', fontsize=13, fontweight='bold')
    ax.set_zlabel('层 z', fontsize=13, fontweight='bold')
    ax.set_title(f'{title}\n加权时间: {tm:.1f}秒', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.view_init(elev=22, azim=-55)

fig.suptitle(f'四种策略三维对比 (最优方案缩短 {(tw_rand - tw_opt) / tw_rand * 100:.1f}%)',
             fontsize=17, fontweight='bold', y=1.01)
plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02, wspace=0.08, hspace=0.20)
fig.savefig(os.path.join(OUT_DIR, '优化vs随机_三维对比.png'), dpi=150, bbox_inches='tight', pad_inches=0.4)
plt.close()
print("Fig1 done")

# ============ FIGURE 2: Metrics bar chart ============
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Bar chart
ax = axes[0]
strategies = ['优化方案', '随机打乱', '逐层填满', '逐列填满']
times = [tw_opt, tw_rand, tw_layer, tw_col]
colors = ['#2196F3', '#CCCCCC', '#FF9800', '#999999']
bars = ax.bar(strategies, times, color=colors, edgecolor='white', width=0.55)
for b, t in zip(bars, times):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f'{t:.1f}s',
            ha='center', fontsize=14, fontweight='bold')
ax.set_ylabel('加权平均出库时间 (秒)', fontsize=13, fontweight='bold')
ax.set_title('四种策略出库时间对比', fontsize=14, fontweight='bold')
# Add improvement annotations
for i, t in enumerate(times[1:], 1):
    imp = (t - tw_opt) / t * 100
    ax.annotate(f'+{imp:.0f}%', xy=(i, t), xytext=(i, t + 3),
                ha='center', fontsize=11, color='#CC3311', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#CC3311'))

# Layer fill pattern comparison
ax = axes[1]
x_pos = np.arange(1, 51)
for label, df_src, c, ls in [('优化方案', df_opt, '#2196F3', '-'),
                               ('逐层填满', df_layer, '#FF9800', '--'),
                               ('逐列填满', df_col, '#999999', ':')]:
    layer_cnt = df_src.groupby('层(z)').size()
    vals = [layer_cnt.get(z, 0) for z in range(1, 51)]
    ax.plot(x_pos, vals, c=c, ls=ls, lw=1.5, label=label, alpha=0.8)
ax.axhline(y=1904, color='green', lw=1, ls=':', alpha=0.5, label='满配1904箱/层')
ax.set_xlabel('层 z', fontsize=13, fontweight='bold')
ax.set_ylabel('填充箱数', fontsize=13, fontweight='bold')
ax.set_title('各层填充量对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)

plt.tight_layout(pad=2.5)
fig.savefig(os.path.join(OUT_DIR, '优化vs随机_指标对比.png'), dpi=150, bbox_inches='tight', pad_inches=0.3)
plt.close()
print("Fig2 done")
print("\nAll done!")
