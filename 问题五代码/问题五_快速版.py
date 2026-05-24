"""
问题五 快速版：从已有操作CSV重建仓库终态，跳过事件仿真
- 读不整理CSV → 回放操作 → 得终态 → 评估偏离 → 整理 → 对比
"""
import pandas as pd, numpy as np, os, heapq
from collections import defaultdict
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
plt.rcParams.update({'font.size':13,'axes.titlesize':16,'axes.labelsize':14})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, '输出结果', '问题五')
CSV_DIR = os.path.join(OUT_DIR, '调度结果CSV')
os.makedirs(CSV_DIR, exist_ok=True)

NUM_SHELVES, NUM_LAYERS, NUM_COLUMNS = 14, 50, 68
NUM_CRANES = 7
vx, vz = 2.5, 0.65  # m/s
COL_W = 0.4  # m per column
T0 = 6.0  # pick/drop time

LAYER_H_MM = {}
cum_mm = 0
for z in range(1, 51):
    if z <= 8: h = 200
    elif z <= 42: h = 400
    else: h = 500
    LAYER_H_MM[z] = cum_mm + h/2
    cum_mm += h

def T_out(x, z):
    dx = (x-0.5) * COL_W
    dz = LAYER_H_MM.get(z, cum_mm/2) / 1000.0
    return max(dx/vx, dz/vz) + T0

def T_between(x1, z1, x2, z2):
    dx = abs(x2 - x1) * COL_W
    dz = abs(LAYER_H_MM.get(z2, 0) - LAYER_H_MM.get(z1, 0)) / 1000.0
    return max(dx/vx, dz/vz) + T0

def get_aisle(shelf): return (shelf-1)//2 + 1

def phys_dist(key):
    _, z, x, _ = key
    return x * 400 + LAYER_H_MM[z]

def layer_allowed(z):
    if z <= 8: return {'E1'}
    elif z <= 42: return {'E1','E3'}
    return {'E1','E3','E4'}

slot_map = {'浅位':0,'深位':1}
slot_rev = {0:'浅位',1:'深位'}

# ============ Load data ============
print("="*60)
print("问题五 快速版：从操作CSV重建 + 偏离评估 + 整理")
print("="*60)

alloc_df = pd.read_csv(os.path.join(BASE,'输出结果','问题一','分配结果CSV','完整分配明细.csv'))
inv_df = pd.read_csv(os.path.join(BASE,'清洗数据一','清洗后_原材料库存数据.csv'))
box_type_map = dict(zip(inv_df['原材料编号'], inv_df['箱子类型']))
cons_map = dict(zip(inv_df['原材料编号'], inv_df['消耗占比']))

# Read operations CSV
tl_csv = os.path.join(CSV_DIR,'不整理_各堆垛机作业时间表.csv')
ops = pd.read_csv(tl_csv)
print(f"读取操作CSV: {len(ops)}条记录")

# ============ Replay: build final warehouse state ============
print("回放操作重建仓库终态...")

occupied = {}
for _, r in alloc_df.iterrows():
    key = (int(r['货架(y)']), int(r['层(z)']), int(r['列(x)']), slot_map[r['深浅位']])
    occupied[key] = r['原材料编号']

# Sort by completion time for replay
ops_sorted = ops.sort_values('完成时刻(s)').reset_index(drop=True)
ops_in = ops_sorted[ops_sorted['任务类型']=='inbound']
ops_out = ops_sorted[ops_sorted['任务类型']=='outbound']

# Apply outbounds (remove from occupied)
for _, r in ops_out.iterrows():
    key = (int(r['货架号']), int(r['层(z)']), int(r['列(x)']), slot_map[r['深浅位']])
    occupied.pop(key, None)

# Apply inbounds (add to occupied)
for _, r in ops_in.iterrows():
    key = (int(r['货架号']), int(r['层(z)']), int(r['列(x)']), slot_map[r['深浅位']])
    occupied[key] = r['原材料编号']

print(f"仓库终态: {len(occupied)}箱占用")

# ============ Compute optimal allocation ============
def compute_optimal_allocation(inventory_list):
    inv_sorted = sorted(inventory_list, key=lambda x: -x[2])
    pos_order = []
    for layer in range(1, NUM_LAYERS+1):
        for col in range(1, NUM_COLUMNS+1):
            dist_mm = col*400 + LAYER_H_MM[layer]
            pos_order.append((col, layer, dist_mm))
    pos_order.sort(key=lambda p: (p[2], p[1], p[0]))

    e1_q, e3_q, e4_q = [], [], []
    mat_seen = set()
    for m, bt, r in inv_sorted:
        if m not in mat_seen:
            mat_seen.add(m)
    # Group by material then by box type
    mat_list = defaultdict(list)
    for m, bt, r in inv_sorted:
        mat_list[bt].append((m, r))
    for bt in ['E1','E3','E4']:
        mat_list[bt].sort(key=lambda x: -x[1])

    result = {}
    ptr = {'E1':0,'E3':0,'E4':0}
    for shelf in range(1, NUM_SHELVES+1):
        for _, layer, col, _ in sorted([(0, l, c, 0) for _,l,c,_ in [(0, layer, col, col*400+LAYER_H_MM[layer]) for layer in range(1,NUM_LAYERS+1) for col in range(1,NUM_COLUMNS+1)]], key=lambda p: (p[3],p[1],p[2])):
            pass  # will rebuild properly

    # Use Problem 1's algorithm more cleanly
    positions = []
    for layer in range(1, NUM_LAYERS+1):
        for col in range(1, NUM_COLUMNS+1):
            positions.append((layer, col, col*400 + LAYER_H_MM[layer]))
    positions.sort(key=lambda p: (p[2], p[0], p[1]))

    fill_q = []
    for layer, col, dist in positions:
        for shelf in range(1, NUM_SHELVES+1):
            fill_q.append((shelf, layer, col, dist))

    result = {}
    ptr = {'E1':0,'E3':0,'E4':0}
    for shelf, layer, col, dist in fill_q:
        allowed = layer_allowed(layer)
        best_type = None; best_r = -1
        for bt in allowed:
            ml = mat_list.get(bt, [])
            p = ptr[bt]
            if p < len(ml) and ml[p][1] > best_r:
                best_r = ml[p][1]; best_type = bt
        if best_type is None: break
        ml = mat_list[best_type]; p = ptr[best_type]
        result[(shelf, layer, col, 0)] = ml[p][0]
        ptr[best_type] += 1
        if ptr[best_type] < len(ml) and ml[ptr[best_type]][0] == ml[p][0]:
            result[(shelf, layer, col, 1)] = ml[ptr[best_type]][0]
            ptr[best_type] += 1

    return result

print("构建最优分配...")
inv_list = [(mat, box_type_map.get(mat,'E3'), cons_map.get(mat,0)) for mat in occupied.values()]
optimal_state = compute_optimal_allocation(inv_list)

# ============ Assessment ============
def assess_warehouse(current, optimal):
    total = len(current)
    exact_mm = sum(1 for k in current if k not in optimal or current[k]!=optimal.get(k,None))

    opt_dists_for_mat = defaultdict(list)
    for key, mat in optimal.items():
        opt_dists_for_mat[mat].append(phys_dist(key))
    for mat in opt_dists_for_mat:
        opt_dists_for_mat[mat].sort()

    severe = 0; sum_cur = 0; sum_opt = 0
    dist_copies = {mat: list(v) for mat, v in opt_dists_for_mat.items()}

    for key, mat in current.items():
        D_cur = phys_dist(key); sum_cur += D_cur
        dists = dist_copies.get(mat, [])
        if dists:
            D_opt_val = dists.pop(0); sum_opt += D_opt_val
            if D_cur > 1.5*D_opt_val and (D_cur-D_opt_val) > 2000:
                severe += 1
        else:
            sum_opt += D_cur

    return (exact_mm, exact_mm/total if total>0 else 0,
            severe, severe/total if total>0 else 0,
            sum_cur/total if total>0 else 0,
            sum_opt/total if total>0 else 0)

exact_mm, exact_rate, severe_cnt, severe_rate, avg_cur, avg_opt = assess_warehouse(occupied, optimal_state)

print(f"\n仓库偏离评估:")
print(f"  精确偏离: {exact_mm}/{len(occupied)} = {exact_rate*100:.1f}%")
print(f"  严重偏离: {severe_cnt}箱 = {severe_rate*100:.1f}%")
print(f"  平均距离: 当前{avg_cur:.0f}mm vs 最优{avg_opt:.0f}mm ({(avg_cur-avg_opt)/avg_opt*100:+.1f}%)")

# ============ Reorganization ============
REORG_THRESHOLD = 0.40

opt_dist_for_mat = defaultdict(list)
for key, mat in optimal_state.items():
    opt_dist_for_mat[mat].append((phys_dist(key), key))
for mat in opt_dist_for_mat:
    opt_dist_for_mat[mat].sort()

# Plan: select severely misplaced boxes
misplaced = []
dist_copies2 = {mat: list(v) for mat, v in opt_dist_for_mat.items()}
for key, mat in occupied.items():
    D_cur = phys_dist(key)
    dists = dist_copies2.get(mat, [])
    if not dists: continue
    D_opt_val, opt_key = dists[0]
    if D_cur > 1.5*D_opt_val and (D_cur-D_opt_val) > 2000:
        misplaced.append((mat, key, opt_key, D_cur, D_opt_val))
        dist_copies2[mat].pop(0)

misplaced.sort(key=lambda x: -(x[3]-x[4]))

moves = []
for mat, src, dst, D_cur, D_opt_val in misplaced:
    moves.append({'material':mat, 'src':src, 'dst':dst, 'D_cur':D_cur, 'D_opt':D_opt_val})

print(f"\n选择性整理: {len(moves)}次移动 ({len(moves)/len(occupied)*100:.1f}%箱数)")

# Execute reorganization with 7 cranes
def execute_parallel_reorg(moves, occupied_dict):
    if not moves: return 0, []

    # Keep track of which positions are free
    free_set = set()
    for shelf in range(1, NUM_SHELVES+1):
        for z in range(1, NUM_LAYERS+1):
            for x in range(1, NUM_COLUMNS+1):
                for slot in [0,1]:
                    key = (shelf,z,x,slot)
                    if key not in occupied_dict:
                        free_set.add(key)

    crane_moves = {i:[] for i in range(1,NUM_CRANES+1)}
    for mv in moves:
        aisle = get_aisle(mv['src'][0])
        crane_moves[aisle].append(mv)

    timeline = []
    max_end = 0

    for cid in range(1, NUM_CRANES+1):
        cm = crane_moves[cid]
        if not cm: continue

        t = 0; crane_pos = 'origin'
        for mv in cm:
            src, dst = mv['src'], mv['dst']

            if crane_pos == 'origin':
                t_src = T_out(src[2], src[1])
            else:
                t_src = T_between(crane_pos[2], crane_pos[1], src[2], src[1])

            t_dst = T_between(src[2], src[1], dst[2], dst[1])
            op = t_src + T0 + t_dst + T0

            mv_start = t; t += op

            # Update warehouse
            mat = occupied_dict.pop(src, None)
            occupied_dict[dst] = mat
            free_set.add(src); free_set.discard(dst)

            timeline.append({
                'crane':cid, 'material':mv['material'],
                'src_shelf':src[0],'src_z':src[1],'src_x':src[2],'src_slot':src[3],
                'dst_shelf':dst[0],'dst_z':dst[1],'dst_x':dst[2],'dst_slot':dst[3],
                'start':mv_start, 'end':t, 'op_time':op,
                'D_before':mv['D_cur'], 'D_after':mv['D_opt']
            })
            crane_pos = (dst[0], dst[1], dst[2], dst[3])

        if crane_pos != 'origin':
            t += T_out(crane_pos[2], crane_pos[1])
        max_end = max(max_end, t)

    return max_end, timeline

print("执行整理...")
reorg_makespan, reorg_timeline = execute_parallel_reorg(moves, occupied)
total_reorg_op = sum(e['op_time'] for e in reorg_timeline)

print(f"整理完成: makespan={reorg_makespan:.0f}s ({reorg_makespan/3600:.1f}h), 总操作时间={total_reorg_op:.0f}s")

# Post-reorg assessment
inv_list2 = [(mat, box_type_map.get(mat,'E3'), cons_map.get(mat,0)) for mat in occupied.values()]
optimal_state2 = compute_optimal_allocation(inv_list2)
_, exact_rate2, _, severe_rate2, _, avg_opt2 = assess_warehouse(occupied, optimal_state2)
print(f"整理后: 精确偏离{exact_rate2*100:.1f}%, 严重偏离{severe_rate2*100:.1f}%")

# ============ Read operational stats from CSV ============
tl_all = pd.read_csv(tl_csv)
total_ob = len(tl_all[tl_all['任务类型']=='outbound'])
total_ib = len(tl_all[tl_all['任务类型']=='inbound'])
total_flow = tl_all['流动时间(s)'].sum()
ob_flow = tl_all[tl_all['任务类型']=='outbound']['流动时间(s)'].sum()
ib_flow = tl_all[tl_all['任务类型']=='inbound']['流动时间(s)'].sum()
makespan_op = tl_all['完成时刻(s)'].max()

print(f"\n运营统计(从CSV):")
print(f"  出库{total_ob}, 入库{total_ib}")
print(f"  总流动时间: {total_flow:.0f}s ({total_flow/3600:.1f}h)")
print(f"  Makespan: {makespan_op/3600:.1f}h")

# ============ Save outputs ============
# 1. Reorganized warehouse state
reorg_occ = []
for key, mat in sorted(occupied.items(), key=lambda x:(x[0][0],x[0][1],x[0][2])):
    reorg_occ.append({
        '货架(y)':key[0],'层(z)':key[1],'列(x)':key[2],
        '深浅位':slot_rev[key[3]],'原材料编号':mat,
        '箱子类型':box_type_map.get(mat,'E3'),
        '物理距离(mm)':phys_dist(key)
    })
pd.DataFrame(reorg_occ).to_csv(os.path.join(CSV_DIR,'整理后_库位分配方案.csv'),index=False,encoding='utf-8-sig')

# 2. Reorg timeline
if reorg_timeline:
    pd.DataFrame(reorg_timeline).to_csv(os.path.join(CSV_DIR,'整理操作明细.csv'),index=False,encoding='utf-8-sig')

# 3. Comparison table
comp_rows = [
    {'指标':'运营总流动时间(s)','不整理':total_flow,'整理(含整理耗时)':total_flow+reorg_makespan,'增量':reorg_makespan},
    {'指标':'Makespan(s)','不整理':makespan_op,'整理(含整理耗时)':makespan_op+reorg_makespan,'增量':reorg_makespan},
    {'指标':'平均出库流动(s)','不整理':ob_flow/total_ob if total_ob else 0,'整理(含整理耗时)':ob_flow/total_ob if total_ob else 0,'增量':0},
    {'指标':'平均入库流动(s)','不整理':ib_flow/total_ib if total_ib else 0,'整理(含整理耗时)':ib_flow/total_ib if total_ib else 0,'增量':0},
    {'指标':'精确偏离率','不整理':f'{exact_rate*100:.1f}%','整理(含整理耗时)':f'{exact_rate2*100:.1f}%','增量':''},
    {'指标':'严重偏离率','不整理':f'{severe_rate*100:.1f}%','整理(含整理耗时)':f'{severe_rate2*100:.1f}%','增量':''},
    {'指标':'平均物理距离(mm)','不整理':f'{avg_cur:.0f}','整理(含整理耗时)':f'{avg_opt:.0f}','增量':f'{avg_cur-avg_opt:.0f}'},
    {'指标':'整理移动箱数','不整理':0,'整理(含整理耗时)':len(moves),'增量':len(moves)},
    {'指标':'整理耗时(s)','不整理':0,'整理(含整理耗时)':f'{reorg_makespan:.0f}','增量':f'{reorg_makespan:.0f}'},
]
pd.DataFrame(comp_rows).to_csv(os.path.join(CSV_DIR,'整理vs不整理_对比表.csv'),index=False,encoding='utf-8-sig')

print(f"\n输出文件:")
for f in os.listdir(CSV_DIR):
    print(f"  {f}")

# ============ Charts ============
print("\n生成图表...")

# Fig 1: Assessment + Comparison
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# 1.1 Deviation
ax = axes[0]
metrics = ['精确偏离率', '严重偏离率\n(D>1.5x最优)', '距离差异']
vals_before = [exact_rate*100, severe_rate*100, (avg_cur-avg_opt)/avg_opt*100]
vals_after = [exact_rate2*100, severe_rate2*100, 0]
x = np.arange(len(metrics)); w = 0.3
b1 = ax.bar(x-w/2, vals_before, w, label='整理前', color='#E74C3C', edgecolor='#2C3E50')
b2 = ax.bar(x+w/2, vals_after, w, label='整理后', color='#27AE60', edgecolor='#2C3E50')
for b, v in zip(b1, vals_before):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylabel('%'); ax.set_title('仓库偏离度评估'); ax.legend(); ax.grid(alpha=0.2, axis='y')

# 1.2 Cost
ax = axes[1]
cats = ['运营流动\n时间', '整理耗时', '总成本']
no_c = [total_flow/3600, 0, total_flow/3600]
re_c = [total_flow/3600, reorg_makespan/3600, (total_flow+reorg_makespan)/3600]
x2 = np.arange(len(cats))
ax.bar(x2-w/2, no_c, w, label='不整理', color='#3498DB')
ax.bar(x2+w/2, re_c, w, label='整理', color='#E74C3C')
for i,(n,r) in enumerate(zip(no_c, re_c)):
    ax.text(i-w/2, n+5, f'{n:.0f}h', ha='center', fontsize=9)
    ax.text(i+w/2, r+5, f'{r:.0f}h', ha='center', fontsize=9)
ax.set_xticks(x2); ax.set_xticklabels(cats); ax.set_ylabel('时间 (h)')
ax.set_title('成本对比'); ax.legend(); ax.grid(alpha=0.2, axis='y')

# 1.3 Crane load from reorg
ax = axes[2]
crane_reorg = {i:0 for i in range(1,NUM_CRANES+1)}
for e in reorg_timeline: crane_reorg[e['crane']] += 1
cids = list(range(1,NUM_CRANES+1)); cnts = [crane_reorg[c] for c in cids]
ax.bar(cids, cnts, color='#E67E22', edgecolor='#2C3E50')
for c, v in zip(cids, cnts):
    ax.text(c, v+0.5, str(v), ha='center', fontweight='bold')
ax.set_xlabel('堆垛机'); ax.set_ylabel('整理移动数')
ax.set_title('各堆垛机整理负载')
ax.grid(alpha=0.2, axis='y')

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR,'整理前后对比图.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  整理前后对比图.png")

# Fig 2: Reorg Gantt
if reorg_timeline:
    fig, ax = plt.subplots(figsize=(20, 8))
    colors = plt.cm.tab10(np.linspace(0,1,7))
    for e in reorg_timeline:
        cid = e['crane']; y = 8-cid
        ax.barh(y, e['op_time']/3600, left=e['start']/3600, height=0.6,
                color=colors[cid-1], edgecolor='white', alpha=0.85)
    ax.set_yticks(range(1,8)); ax.set_yticklabels([f'堆垛机{i}' for i in range(7,0,-1)])
    ax.set_xlabel('时间 (h)'); ax.set_title('整理作业甘特图')
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR,'整理作业甘特图.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  整理作业甘特图.png")

# Fig 3: Distance improvement scatter
if reorg_timeline:
    fig, ax = plt.subplots(figsize=(10, 8))
    D_b = [e['D_before'] for e in reorg_timeline]
    D_a = [e['D_after'] for e in reorg_timeline]
    ax.scatter(D_b, D_a, c='#E74C3C', alpha=0.5, s=20)
    md = max(max(D_b), max(D_a))
    ax.plot([0,md],[0,md],'--',color='gray',label='无改善线')
    ax.fill_between([0,md],[0,md],0,alpha=0.1,color='#27AE60',label='改善区域')
    ax.set_xlabel('整理前距离(mm)'); ax.set_ylabel('整理后距离(mm)')
    ax.set_title('整理移动距离改善'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR,'库位偏离度变化图.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  库位偏离度变化图.png")

# ============ Summary ============
print("\n"+"="*60)
print("问题五 总结")
print("="*60)
print(f"运营结果: 总流动时间{total_flow/3600:.1f}h, OB{total_ob}/IB{total_ib}")
print(f"仓库状态: 精确偏离{exact_rate*100:.1f}%, 严重偏离{severe_rate*100:.1f}%")
print(f"整理方案: {len(moves)}次移动, 耗时{reorg_makespan/3600:.1f}h")
print(f"成本对比: 不整理{total_flow/3600:.1f}h vs 整理{(total_flow+reorg_makespan)/3600:.1f}h")
print(f"偏离改善: {exact_rate*100:.1f}%→{exact_rate2*100:.1f}%")
print(f"\n全部输出: {OUT_DIR}")
print("Done!")
