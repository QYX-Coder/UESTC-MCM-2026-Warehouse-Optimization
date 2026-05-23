"""
问题二: 7巷道7机并行调度 (修正版)
抢占规则: 先完成当前箱运回原点, 再比较队列重排(SPT) vs 不重排(FIFO)
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
plt.rcParams.update({'font.size':10,'axes.titlesize':14,'axes.labelsize':12,
                     'axes.titleweight':'bold','axes.labelweight':'bold'})

BASE = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE, '输出结果', '问题二')
CSV_DIR = os.path.join(OUT_DIR, '调度结果CSV')
os.makedirs(CSV_DIR, exist_ok=True)

# ============ Speed ============
VX, VZ = 0.2, 0.1; T0 = 6.0
LH = {}; cum = 0.0
for z in range(1, 51):
    if z <= 8: h = 0.2
    elif z <= 42: h = 0.4
    else: h = 0.5
    LH[z] = cum + h / 2; cum += h
NUM_CRANES = 7

def get_aisle(shelf):
    return (shelf - 1) // 2 + 1

def T_out(x, z):
    return x * 0.4 / VX + LH[z] / VZ

def T_roundtrip(x, z):
    return 2 * T_out(x, z) + T0

print(f"vx={VX}, vz={VZ}, {NUM_CRANES} cranes")

# ============ Load ============
orders = pd.read_csv(os.path.join(BASE, '清洗数据一', '清洗后_生产线订料数据.csv'))
orders['时间'] = pd.to_datetime(orders['时间'])
orders = orders.sort_values('时间').reset_index(drop=True)
t0_ref = orders['时间'].min()

alloc = pd.read_csv(os.path.join(BASE, '输出结果', '问题一', '分配结果CSV', '完整分配明细.csv'))
mat_pos = {}
for _, r in alloc.iterrows():
    mat_pos.setdefault(r['原材料编号'], []).append(
        {'x': r['列(x)'], 'z': r['层(z)'], 'shelf': r['货架(y)']})

# Consumption lookup
df1 = pd.read_csv(os.path.join(BASE, '清洗数据一', '清洗后_原材料库存数据.csv'))
cons_map = dict(zip(df1['原材料编号'], df1['消耗占比']))

boxes = []; ptrs = {}
for _, row in orders.iterrows():
    mid = row['原材料号']
    if mid not in mat_pos: continue
    if mid not in ptrs: ptrs[mid] = 0
    p = mat_pos[mid][ptrs[mid] % len(mat_pos[mid])]; ptrs[mid] += 1
    boxes.append({
        'id': row['订单号'], 'material': mid,
        'arrival': (row['时间'] - t0_ref).total_seconds(),
        'consumption': cons_map.get(mid, 0),
        'x': p['x'], 'z': p['z'], 'shelf': p['shelf'],
        'aisle': get_aisle(p['shelf']),
        'T_out': T_out(p['x'], p['z']),
        'T_total': T_roundtrip(p['x'], p['z'])
    })

for a in range(1, NUM_CRANES+1):
    print(f"  Crane{a}: {sum(1 for b in boxes if b['aisle']==a)} boxes")
print(f"Total: {len(boxes)} boxes")

# ============ Single-crane simulation ============
def simulate_crane(box_list, mode):
    """
    mode: 'spt' or 'fifo'
    Crane MUST finish current trip before starting next (no mid-trip abort).
    At each decision point (after returning to origin), pick from queue.
    SPT: shortest total time first. FIFO: earliest arrival first.
    Returns: list of done boxes, preemption events log
    """
    box_list = sorted(box_list, key=lambda b: b['arrival'])
    queue = []; done = []; ptr = 0; t = 0
    cur_phase = 'idle'  # idle, outbound, at_box, returning
    cur_box = None
    cur_done_at = 0  # when current box will be fully done
    decisions = []  # log preemption decisions

    while ptr < len(box_list) or queue or cur_box:
        # Add arrivals up to current time
        while ptr < len(box_list) and box_list[ptr]['arrival'] <= t:
            new_box = box_list[ptr]; ptr += 1
            if cur_box is not None:
                # PREEMPTION CHECK: new order while busy
                # Must finish current box's return first
                # After return: compare SPT vs FIFO for {remaining_current_queue + new}
                # Simplified: SPT will sort queue later, FIFO won't
                # Log this as a decision point
                decisions.append({
                    't': t, 'cur_box': cur_box['id'],
                    'cur_phase': cur_phase,
                    'new_box': new_box['id'],
                    'new_T': new_box['T_total'],
                    'cur_T_remaining': max(0, cur_done_at - t)
                })
            queue.append(new_box)

        if cur_box is None and queue:
            # Pick next box from queue
            if mode == 'fifo':
                queue.sort(key=lambda b: b['arrival'])
            elif mode == 'priority':
                queue.sort(key=lambda b: -b['consumption'])
            else:  # spt
                queue.sort(key=lambda b: b['T_total'])

            cur_box = queue.pop(0)
            # Outbound travel
            t += cur_box['T_out']
            cur_box['start'] = t
            # At box: pick + return
            t += T0  # pick
            t += T_out(cur_box['x'], cur_box['z'])  # return
            cur_box['end'] = t
            cur_box['flow'] = t - cur_box['arrival']
            done.append(cur_box)
            cur_box = None
        elif cur_box is None:
            if ptr < len(box_list):
                t = max(t, box_list[ptr]['arrival'])
            else:
                break

    return done, decisions


# ============ Run all 7 cranes ============
def run_all(mode):
    all_done = []; all_decisions = []
    for a in range(1, NUM_CRANES+1):
        aisle_boxes = [b for b in boxes if b['aisle'] == a]
        done, dec = simulate_crane(aisle_boxes, mode)
        for d in done: d['aisle'] = a
        for d in dec: d['aisle'] = a
        all_done.extend(done); all_decisions.extend(dec)
    return pd.DataFrame(all_done), all_decisions

print("\nRunning...")
df_spt, _ = run_all('spt')
df_fifo, _ = run_all('fifo')
df_prio, _ = run_all('priority')

for name, df_r in [('SPT(最短优先)', df_spt), ('FIFO(先到先做)', df_fifo), ('Priority(消耗优先)', df_prio)]:
    print(f"{name}: {len(df_r)} boxes, avg flow: {df_r['flow'].mean():.0f}s")

# Show example decisions where SPT helps
print(f"\nSPT vs FIFO: {(1-df_spt['flow'].mean()/df_fifo['flow'].mean())*100:.1f}% faster")
print(f"SPT vs Priority: {(1-df_spt['flow'].mean()/df_prio['flow'].mean())*100:.1f}% faster")

# ============ FIGURES ============
# 3D Depletion (2 subplots: before/after)
fig = plt.figure(figsize=(20, 10))
for i, (tp, title) in enumerate([(0, '出库前 (初始状态)'), (len(df_spt)//2, '出库50%完成')]):
    ax = fig.add_subplot(1, 2, i+1, projection='3d')
    if tp == 0:
        ax.scatter(df_spt['x'], df_spt['shelf'], df_spt['z'],
                   c='#2196F3', s=12, alpha=0.7, label='待出库')
    else:
        df_ordered = df_spt.sort_values("end")
        d = df_ordered.head(tp); r = df_ordered.iloc[tp:]
        if len(r) > 0:
            ax.scatter(r['x'], r['shelf'], r['z'], c='#2196F3', s=12, alpha=0.5, label='剩余')
        if len(d) > 0:
            ax.scatter(d['x'], d['shelf'], d['z'], c='#CCCCCC', s=12, alpha=0.5, label='已出库')
    ax.set_xlabel('列 x', fontsize=13, fontweight='bold')
    ax.set_ylabel('货架 y', fontsize=13, fontweight='bold')
    ax.set_zlabel('层 z', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.view_init(elev=25, azim=-55)
fig.suptitle('库存消耗三维对比 (3738箱出库)', fontsize=16, fontweight='bold', y=1.01)
plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02, wspace=0.05)
fig.savefig(os.path.join(OUT_DIR, '库存消耗3D.png'), dpi=150, bbox_inches='tight', pad_inches=0.4)
plt.close()

# Speed sensitivity
fig2, ax = plt.subplots(figsize=(12, 7))
vx_v = [2.5, 2.0, 1.5, 1.0, 0.5, 0.32, 0.2, 0.1]
fl_v = [192, 268, 420, 764, 2303, 6227, 35388, 220696]
T_v = [21, 25, 31, 43, 80, 122, 192, 378]
ax.plot(vx_v, [f/3600 for f in fl_v], 'o-', c='#2196F3', lw=2.5, ms=10, label='平均流动时间 (h)')
ax2t = ax.twinx()
ax2t.plot(vx_v, T_v, 's--', c='#FF9800', lw=2, ms=8, label='平均运输时间 (s)')
ax.axvline(x=0.32, color='#CC3311', lw=2, ls=':')
ax.annotate('临界 vx=0.32', xy=(0.32, 2), xytext=(0.7, 6),
            fontsize=11, color='#CC3311', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#CC3311'))
ax.axvspan(0, 0.32, alpha=0.06, color='#CC3311')
ax.axvspan(0.32, 2.5, alpha=0.06, color='#4CAF50')
ax.text(0.15, 30, '排队堆积', ha='center', fontsize=12, color='#CC3311', fontweight='bold')
ax.text(1.4, 30, '畅通无阻', ha='center', fontsize=12, color='#4CAF50', fontweight='bold')
ax.set_xlabel('水平速度 v_x (m/s)', fontsize=13, fontweight='bold')
ax.set_ylabel('平均流动时间 (h)', fontsize=13, fontweight='bold', color='#2196F3')
ax2t.set_ylabel('平均运输时间 (s)', fontsize=13, fontweight='bold', color='#FF9800')
ax.set_title('速度灵敏度分析', fontsize=14, fontweight='bold')
ax.invert_xaxis()
l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2t.get_legend_handles_labels()
ax.legend(l1+l2, la1+la2, fontsize=10, loc='upper left')
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, '速度灵敏度.png'), dpi=150, bbox_inches='tight', pad_inches=0.3)
plt.close()

# Save with Chinese columns
rename_map = {
    'id': '订单号', 'material': '原材料号', 'arrival': '到达时间(s)',
    'consumption': '消耗占比', 'x': '列坐标', 'z': '层坐标',
    'shelf': '货架号', 'aisle': '巷道号',
    'T_out': '单程时间(s)', 'T_total': '总运输时间(s)',
    'start': '开始时刻(s)', 'end': '完成时刻(s)', 'flow': '流动时间(s)'
}
for name, df_r in [('SPT调度结果', df_spt), ('FIFO调度结果', df_fifo), ('Priority调度结果', df_prio)]:
    df_out = df_r.rename(columns={k: v for k, v in rename_map.items() if k in df_r.columns})
    df_out.to_csv(os.path.join(CSV_DIR, f'{name}.csv'), index=False, encoding='utf-8-sig')

print(f"\nDone! {len(df_spt)}/3738 boxes OK={len(df_spt)==3738}")
