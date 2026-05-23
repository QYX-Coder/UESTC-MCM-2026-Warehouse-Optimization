"""
问题一：库位初始分配优化 (物理距离修正版)
填充排序: max(x*400mm, H(z)mm) 从小到大
层高: 1-8层200mm, 9-42层400mm, 43-50层500mm
列宽: 400mm
"""
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for fn in ['Microsoft YaHei', 'SimSun', 'SimHei']:
    try:
        fm.findfont(fn, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [fn, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue

NUM_SHELVES, NUM_LAYERS, NUM_COLUMNS = 14, 50, 68
COL_WIDTH = 400  # mm per column

# Cumulative layer center heights (mm)
LAYER_H = {}
cum = 0
for z in range(1, 51):
    if z <= 8: h = 200
    elif z <= 42: h = 400
    else: h = 500
    LAYER_H[z] = cum + h / 2
    cum += h

BASE = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE, '输出结果', '问题一')
os.makedirs(OUT_DIR, exist_ok=True)

# ============ 1. Load data ============
df1 = pd.read_csv(os.path.join(BASE, '清洗数据一', '清洗后_原材料库存数据.csv'))
df1['消耗占比'] = pd.to_numeric(df1['消耗占比'], errors='coerce')
df1['库存数量/箱'] = pd.to_numeric(df1['库存数量/箱'], errors='coerce').astype(int)

print("=" * 60)
print("Problem 1: Physical distance optimization")
print(f"Column width: {COL_WIDTH}mm, Layer heights: 200/400/500mm")
print(f"Sort metric: x*{COL_WIDTH} + H(z) (sum, sequential)")
print("=" * 60)

# ============ 2. Material groups ============
groups = {
    'E1': df1[df1['箱子类型'] == 'E1'].sort_values('消耗占比', ascending=False),
    'E3': df1[df1['箱子类型'] == 'E3'].sort_values('消耗占比', ascending=False),
    'E4': df1[df1['箱子类型'] == 'E4'].sort_values('消耗占比', ascending=False),
}

remaining = {}
for bt in ['E1', 'E3', 'E4']:
    for _, r in groups[bt].iterrows():
        remaining[r['原材料编号']] = int(r['库存数量/箱'])

queues = {bt: groups[bt]['原材料编号'].tolist() for bt in ['E1', 'E3', 'E4']}
ptr = {'E1': 0, 'E3': 0, 'E4': 0}

for bt in ['E1', 'E3', 'E4']:
    print(f"  {bt}: {len(groups[bt])} types, {groups[bt]['库存数量/箱'].sum()} boxes")
print(f"  Total: {sum(g['库存数量/箱'].sum() for g in groups.values())} boxes")

def layer_allowed(z):
    if z <= 8: return ['E1']
    elif z <= 42: return ['E1', 'E3']
    return ['E1', 'E3', 'E4']

def pick_best(allowed):
    best_bt, best_ratio, best_id = None, -1, None
    for bt in allowed:
        while ptr[bt] < len(queues[bt]) and remaining[queues[bt][ptr[bt]]] <= 0:
            ptr[bt] += 1
        if ptr[bt] < len(queues[bt]):
            mid = queues[bt][ptr[bt]]
            r = groups[bt][groups[bt]['原材料编号'] == mid]['消耗占比'].values[0]
            if r > best_ratio:
                best_ratio, best_bt, best_id = r, bt, mid
    return best_bt, best_id

# ============ 3. Fill: unified physical distance col*400 + H(layer) (sum) ============
pos_order = []
for layer in range(1, NUM_LAYERS + 1):
    for col in range(1, NUM_COLUMNS + 1):
        dist_mm = col * COL_WIDTH + LAYER_H[layer]
        pos_order.append((col, layer, dist_mm))
pos_order.sort(key=lambda p: (p[2], p[1], p[0]))

fill_queue = []
for col, layer, dist_mm in pos_order:
    for shelf in range(1, NUM_SHELVES + 1):
        fill_queue.append((col, layer, shelf, dist_mm))

print(f"Fill queue: {len(fill_queue)} positions (unified physical distance)")

# ============ 4. Allocate ============
allocation = []
total = 0
last_report = 0

for col, layer, shelf, dist_mm in fill_queue:
    allowed = layer_allowed(layer)
    bt, mat_id = pick_best(allowed)
    if bt is None: continue

    qty = remaining[mat_id]
    ratio = groups[bt][groups[bt]['原材料编号'] == mat_id]['消耗占比'].values[0]
    slot_base = f'{shelf:02d}_{layer:02d}_{col:02d}'
    xz_dist = col + layer  # keep for reference

    if qty >= 2:
        allocation.extend([
            {'货架编号': f'{shelf:02d}', '货架(y)': shelf, '层(z)': layer, '列(x)': col,
             '深浅位': '浅位', '原材料编号': mat_id, '箱子类型': bt,
             '消耗占比': ratio, '距离(x+z)': xz_dist, '物理距离(mm)': dist_mm,
             '库位编号': f'{slot_base}_01'},
            {'货架编号': f'{shelf:02d}', '货架(y)': shelf, '层(z)': layer, '列(x)': col,
             '深浅位': '深位', '原材料编号': mat_id, '箱子类型': bt,
             '消耗占比': ratio, '距离(x+z)': xz_dist, '物理距离(mm)': dist_mm,
             '库位编号': f'{slot_base}_02'}
        ])
        remaining[mat_id] -= 2; total += 2
    else:
        allocation.append(
            {'货架编号': f'{shelf:02d}', '货架(y)': shelf, '层(z)': layer, '列(x)': col,
             '深浅位': '浅位', '原材料编号': mat_id, '箱子类型': bt,
             '消耗占比': ratio, '距离(x+z)': xz_dist, '物理距离(mm)': dist_mm,
             '库位编号': f'{slot_base}_01'}
        )
        remaining[mat_id] -= 1; total += 1

    if total - last_report >= 20000:
        print(f"  {total}/65000 ({total/65000*100:.1f}%) pos=({col},{layer},y={shelf}) dist={dist_mm:.0f}mm")
        last_report = total

df_alloc = pd.DataFrame(allocation)

# ============ 5. Statistics ============
print(f"\nAllocated: {len(df_alloc)} boxes")
shallow = (df_alloc['深浅位'] == '浅位').sum()
deep = (df_alloc['深浅位'] == '深位').sum()
print(f"Shallow: {shallow}, Deep: {deep}, Diff: {shallow - deep}")
print(f"Phys dist (mm): min={df_alloc['物理距离(mm)'].min():.0f}, "
      f"max={df_alloc['物理距离(mm)'].max():.0f}, mean={df_alloc['物理距离(mm)'].mean():.0f}")

for s in range(1, NUM_SHELVES + 1):
    cnt = len(df_alloc[df_alloc['货架(y)'] == s])
    print(f"  Shelf {s:02d}: {cnt} boxes")

# Layer fill
print("\nLayer fill (column ranges):")
for z in range(1, 51):
    sub = df_alloc[df_alloc['层(z)'] == z]
    if len(sub) > 0:
        print(f"  L{z:2d}: {len(sub):4d} boxes, cols {sub['列(x)'].min()}-{sub['列(x)'].max()}, "
              f"types={sub['箱子类型'].unique()}")

# ============ 6. Save CSV ============
csv_dir = os.path.join(OUT_DIR, '分配结果CSV')
os.makedirs(csv_dir, exist_ok=True)

df_alloc.to_csv(os.path.join(csv_dir, '完整分配明细.csv'), index=False, encoding='utf-8-sig')

shelf_s = df_alloc.groupby(['货架编号', '货架(y)']).agg(
    分配箱数=('原材料编号', 'count'),
    E1箱=('箱子类型', lambda x: (x=='E1').sum()),
    E3箱=('箱子类型', lambda x: (x=='E3').sum()),
    E4箱=('箱子类型', lambda x: (x=='E4').sum()),
    平均物理距离=('物理距离(mm)', 'mean'),
).reset_index()
shelf_s.to_csv(os.path.join(csv_dir, '按货架汇总.csv'), index=False, encoding='utf-8-sig')

# Time calculation
VX_E, VZ_E = 3.0, 0.75
VX_L, VZ_L = 2.3, 0.58
T_FORK, T_OTHER = 3.0, 3.0

def calc_time(row):
    Dx = row['列(x)'] * COL_WIDTH / 1000  # mm -> m
    Dz = LAYER_H[row['层(z)']] / 1000      # mm -> m
    t1 = Dx / VX_E + Dz / VZ_E
    t2 = Dx / VX_L + Dz / VZ_L
    return t1 + t2 + T_FORK + T_OTHER

times = np.array([calc_time(r) for _, r in df_alloc.iterrows()])
w = df_alloc['消耗占比'] / df_alloc['消耗占比'].sum()
w_mean = np.average(times, weights=w)
a_mean = times.mean()

print(f"\nRetrieval time (linear): weighted={w_mean:.2f}s, mean={a_mean:.2f}s")

time_df = df_alloc[['库位编号', '原材料编号', '箱子类型', '列(x)', '层(z)', '物理距离(mm)']].copy()
time_df['取货时间(s)'] = [r['列(x)']*COL_WIDTH/1000/VX_E + LAYER_H[r['层(z)']]/1000/VZ_E
                         for _, r in df_alloc.iterrows()]
time_df['送货时间(s)'] = [r['列(x)']*COL_WIDTH/1000/VX_L + LAYER_H[r['层(z)']]/1000/VZ_L
                         for _, r in df_alloc.iterrows()]
time_df['单次出库时间(s)'] = time_df['取货时间(s)'] + time_df['送货时间(s)'] + T_FORK + T_OTHER
time_df.to_csv(os.path.join(csv_dir, '出库时间明细.csv'), index=False, encoding='utf-8-sig')

print(f"CSV saved to: {csv_dir}")
print("\nDone!")
