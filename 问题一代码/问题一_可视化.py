"""
问题一 可视化：三维结构图 + 切面图 (物理距离版)
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
df = pd.read_csv(os.path.join(CSV_DIR, '完整分配明细.csv'))
N_S, N_L, N_C = 14, 50, 68
C_MAP = {'E1': '#2196F3', 'E3': '#FF9800', 'E4': '#4CAF50'}
L_MAP = {'E1': 'E1箱', 'E3': 'E3箱', 'E4': 'E4箱'}

# ============ FIGURE 1: 3D ============
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
for bt, c in C_MAP.items():
    sub = df[df['箱子类型'] == bt]
    if len(sub) > 0:
        ax.scatter(sub['列(x)'], sub['货架(y)'], sub['层(z)'],
                   c=c, s=2, alpha=0.5, label=L_MAP[bt], rasterized=True)
for z in [8.5, 42.5]:
    ax.plot([1, N_C], [N_S/2]*2, [z, z], 'gray', lw=1, ls=':', alpha=0.5)
ax.set_xlabel('列 x (深度)', fontsize=13, fontweight='bold')
ax.set_ylabel('货架 y', fontsize=13, fontweight='bold')
ax.set_zlabel('层 z (高度)', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='z', labelsize=14)
ax.set_title(f'自动立体库 库位分配三维结构图 (物理距离排序)\n(列宽400mm, 层高200/400/500mm, {len(df)}箱)',
             fontsize=20, fontweight='bold')
ax.legend(fontsize=15, loc='upper right', title='箱型', title_fontsize=16,markerscale=6)
ax.view_init(elev=22, azim=-55)
plt.subplots_adjust(left=0.02, right=0.95, top=0.95, bottom=0.02)
fig.savefig(os.path.join(OUT_DIR, '仓库三维结构图.png'), dpi=150, bbox_inches='tight', pad_inches=0.4)
plt.close()
print("Fig1 done")

# ============ FIGURE 2: Cross sections ============
fig, axes = plt.subplots(2, 2, figsize=(20, 17))

# 2a: Shelf 1 x-z
ax = axes[0, 0]
g = np.full((N_L, N_C), '', dtype=object)
for _, r in df[df['货架(y)'] == 1].iterrows():
    g[r['层(z)']-1, r['列(x)']-1] = r['箱子类型']
for z in range(N_L):
    for x in range(N_C):
        c = C_MAP.get(g[z,x], '#FFFFFF')
        ax.add_patch(plt.Rectangle((x+0.5, z+0.5), 1, 1, fc=c, ec='white', lw=0.12))
ax.set_xlim(0.5, N_C+0.5); ax.set_ylim(0.5, N_L+0.5)
ax.axhline(8.5, color='#0077BB', lw=2, ls='--'); ax.axhline(42.5, color='#CC3311', lw=2, ls='--')
ax.set_xlabel('列 x', fontsize=13, fontweight='bold')
ax.set_ylabel('层 z', fontsize=13, fontweight='bold')
ax.set_title('货架1 切面 (x-z平面)\n蓝=E1 | 橙=E3 | 绿=E4, 竖墙边界在col 47', fontsize=13, fontweight='bold')
ax.legend(handles=[plt.plot([],[],marker='s',c=C_MAP[b],ls='',markersize=12,label=L_MAP[b])[0] for b in C_MAP], fontsize=10)

# 2b: Layer 1 x-y
ax = axes[0, 1]
g = np.full((N_S, N_C), '', dtype=object)
for _, r in df[df['层(z)'] == 1].iterrows():
    g[r['货架(y)']-1, r['列(x)']-1] = r['箱子类型']
for y in range(N_S):
    for x in range(N_C):
        c = C_MAP.get(g[y,x], '#FFFFFF')
        ax.add_patch(plt.Rectangle((x+0.5, y+0.5), 1, 1, fc=c, ec='white', lw=0.12))
ax.set_xlim(0.5, N_C+0.5); ax.set_ylim(0.5, N_S+0.5)
ax.set_xlabel('列 x', fontsize=13, fontweight='bold')
ax.set_ylabel('货架 y', fontsize=13, fontweight='bold')
ax.set_title('第1层 切面 (x-y平面, E1区)\n14货架 x 47列填满, cols 48-68空', fontsize=13, fontweight='bold')

# 2c: Column 1 y-z
ax = axes[1, 0]
g = np.full((N_L, N_S), '', dtype=object)
for _, r in df[df['列(x)'] == 1].iterrows():
    g[r['层(z)']-1, r['货架(y)']-1] = r['箱子类型']
for z in range(N_L):
    for y in range(N_S):
        c = C_MAP.get(g[z,y], '#FFFFFF')
        ax.add_patch(plt.Rectangle((y+0.5, z+0.5), 1, 1, fc=c, ec='white', lw=0.12))
ax.set_xlim(0.5, N_S+0.5); ax.set_ylim(0.5, N_L+0.5)
ax.axhline(8.5, color='#0077BB', lw=2, ls='--'); ax.axhline(42.5, color='#CC3311', lw=2, ls='--')
ax.set_xlabel('货架 y', fontsize=13, fontweight='bold')
ax.set_ylabel('层 z', fontsize=13, fontweight='bold')
ax.set_title('第1列 切面 (y-z平面, x=1最近列)\n列1全50层填满, 14货架均匀', fontsize=13, fontweight='bold')

# 2d: Layer 25 x-y (E3 zone)
ax = axes[1, 1]
g = np.full((N_S, N_C), '', dtype=object)
for _, r in df[df['层(z)'] == 25].iterrows():
    g[r['货架(y)']-1, r['列(x)']-1] = r['箱子类型']
for y in range(N_S):
    for x in range(N_C):
        c = C_MAP.get(g[y,x], '#FFFFFF')
        ax.add_patch(plt.Rectangle((x+0.5, y+0.5), 1, 1, fc=c, ec='white', lw=0.12))
ax.set_xlim(0.5, N_C+0.5); ax.set_ylim(0.5, N_S+0.5)
ax.set_xlabel('列 x', fontsize=13, fontweight='bold')
ax.set_ylabel('货架 y', fontsize=13, fontweight='bold')
ax.set_title('第25层 切面 (x-y平面, E3区)\n同第1层: 14货架 x 47列填满', fontsize=13, fontweight='bold')

fig.suptitle('库位分配切面图 (物理距离排序, 四个视角)', fontsize=17, fontweight='bold', y=0.99)
plt.tight_layout(pad=2.5)
fig.savefig(os.path.join(OUT_DIR, '库位分配切面图.png'), dpi=150, bbox_inches='tight', pad_inches=0.3)
plt.close()
print("Fig2 done")
print("\nAll done!")
