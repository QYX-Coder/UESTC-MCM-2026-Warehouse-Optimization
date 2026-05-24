"""
问题三：入库-出库协同调度优化
- 7巷道7机并行复合作业
- 目标: min Σ(flow_time) = Σ(完成时刻-到达时刻)
- 入库: 单箱SPT, T_store=T_out+T0, 放到货位即完成
- 出库: SPT, T_total=2*T_out+T0, 回原点完成
- 参照问题二假设参数表
"""
import pandas as pd, numpy as np, os, heapq, copy
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
OUT_DIR = os.path.join(BASE, '输出结果', '问题三')
CSV_DIR = os.path.join(OUT_DIR, '调度结果CSV')
os.makedirs(CSV_DIR, exist_ok=True)

# ============ Physical constants (参照问题二假设参数表) ============
VX, VZ = 0.2, 0.1  # m/s
T0 = 6.0            # s, 固定操作时间
NUM_SHELVES, NUM_LAYERS, NUM_COLUMNS = 14, 50, 68
NUM_CRANES = 7
COL_WIDTH = 0.4  # m

# Layer heights (参照问题二)
LH = {}; cum = 0.0
for z in range(1, 51):
    if z <= 8: h = 0.2
    elif z <= 42: h = 0.4
    else: h = 0.5
    LH[z] = cum + h / 2; cum += h

def get_aisle(shelf):
    return (shelf - 1) // 2 + 1

def layer_allowed(z):
    if z <= 8: return ['E1']
    elif z <= 42: return ['E1', 'E3']
    return ['E1', 'E3', 'E4']

def T_out(x, z):
    """单程：原点→(x,z)"""
    return x * COL_WIDTH / VX + LH[z] / VZ

def T_between(x1, z1, x2, z2):
    """两位间移动时间（顺序运动）"""
    return abs(x2 - x1) * COL_WIDTH / VX + abs(LH[z2] - LH[z1]) / VZ

# ============ Data loading ============
print("=" * 60)
print("问题三：出入库协同调度优化")
print(f"vx={VX}, vz={VZ}, T0={T0}s, {NUM_CRANES}台堆垛机")
print("=" * 60)

# 1. Load inventory & Problem 1 allocation
alloc_df = pd.read_csv(os.path.join(BASE, '输出结果', '问题一', '分配结果CSV', '完整分配明细.csv'))
inv_df = pd.read_csv(os.path.join(BASE, '清洗数据一', '清洗后_原材料库存数据.csv'))
cons_map = dict(zip(inv_df['原材料编号'], inv_df['消耗占比']))

# 2. Load outbound orders (Problem 2 data)
out_orders = pd.read_csv(os.path.join(BASE, '清洗数据一', '清洗后_生产线订料数据.csv'))
out_orders['时间'] = pd.to_datetime(out_orders['时间'])
out_orders = out_orders.sort_values('时间').reset_index(drop=True)

# 3. Load inbound orders (Table 3)
in_orders = pd.read_csv(os.path.join(BASE, '清洗数据一', '清洗后_入库材料数据.csv'))
in_orders['入库时间'] = pd.to_datetime(in_orders['入库时间'])
in_orders = in_orders.sort_values('入库时间').reset_index(drop=True)

# Unified time reference (earliest among all events)
t0_ref = min(out_orders['时间'].min(), in_orders['入库时间'].min())
print(f"t0_ref = {t0_ref}")

# ============ Warehouse state initialization ============
# occupied[shelf_0idx][z][x][slot] = material_id or None
occupied = [[[[None for _ in range(2)] for _ in range(NUM_COLUMNS + 1)]
             for _ in range(NUM_LAYERS + 1)] for _ in range(NUM_SHELVES)]

# mat_positions[material_id] = [{'shelf':, 'z':, 'x':, 'slot':, 'T_out':}, ...]
mat_positions = {}

slot_map = {'浅位': 0, '深位': 1}
for _, r in alloc_df.iterrows():
    shelf = int(r['货架(y)'])
    z = int(r['层(z)'])
    x = int(r['列(x)'])
    slot = slot_map[r['深浅位']]
    mid = r['原材料编号']
    occupied[shelf-1][z][x][slot] = mid
    t_out = T_out(x, z)
    mat_positions.setdefault(mid, []).append(
        {'shelf': shelf, 'z': z, 'x': x, 'slot': slot, 'T_out': t_out})

alloc_count = len(alloc_df)
print(f"初始仓库: {alloc_count}箱占位")

# Build empty position structures
empty_by_aisle = {a: [] for a in range(1, NUM_CRANES + 1)}
empty_set = set()
for shelf in range(1, NUM_SHELVES + 1):
    aisle = get_aisle(shelf)
    for z in range(1, NUM_LAYERS + 1):
        for x in range(1, NUM_COLUMNS + 1):
            for slot in [0, 1]:
                if occupied[shelf-1][z][x][slot] is None:
                    pos_tuple = (shelf, z, x, slot)
                    empty_set.add(pos_tuple)
                    empty_by_aisle[aisle].append(pos_tuple)

# Sort each aisle's empty list by T_out
for a in range(1, NUM_CRANES + 1):
    empty_by_aisle[a].sort(key=lambda p: T_out(p[2], p[1]))

empty_count = len(empty_set)
print(f"初始空位: {empty_count} (预期30200)")
print(f"各巷道空位: {[(a, len(empty_by_aisle[a])) for a in range(1,8)]}")

# ============ Prepare outbound orders ============
outbound_list = []
for _, r in out_orders.iterrows():
    mid = r['原材料号']
    arrival = (r['时间'] - t0_ref).total_seconds()
    outbound_list.append({
        'id': r['订单号'], 'material': mid,
        'box_type': r['箱子类型'], 'arrival': arrival,
        'assigned_pos': None  # filled at execution time
    })
print(f"出库订单: {len(outbound_list)}单")

# ============ Prepare inbound orders ============
inbound_batches = []
for _, r in in_orders.iterrows():
    arrival = (r['入库时间'] - t0_ref).total_seconds()
    inbound_batches.append({
        'batch_id': r['入库批次'], 'material': r['原材料编号'],
        'box_type': r['箱子类型'], 'box_count': int(r['入库数量/箱']),
        'arrival': arrival
    })
total_inbound_boxes = sum(b['box_count'] for b in inbound_batches)
print(f"入库批次: {len(inbound_batches)}批, 共{total_inbound_boxes}箱")

# ============ Helper: find nearest empty position ============
def find_nearest_empty_global(box_type):
    """全局搜索最近空位（用于入库到达时的预览排序），返回(shelf,z,x,slot)或None"""
    best = None; best_t = float('inf')
    for a in range(1, NUM_CRANES + 1):
        for pos in empty_by_aisle[a]:
            shelf, z, x, slot = pos
            if pos not in empty_set: continue
            if box_type not in layer_allowed(z): continue
            t = T_out(x, z)
            if t < best_t:
                best_t = t; best = pos
            break
    return best

def find_nearest_empty_aisle(box_type, aisle):
    """巷道内搜索最近空位（用于执行时），返回(shelf,z,x,slot)或None"""
    best = None; best_t = float('inf')
    for pos in empty_by_aisle[aisle]:
        shelf, z, x, slot = pos
        if pos not in empty_set: continue
        if box_type not in layer_allowed(z): continue
        t = T_out(x, z)
        if t < best_t:
            best_t = t; best = pos
        break
    return best

# ============ Crane class ============
class Crane:
    def __init__(self, cid):
        self.id = cid
        self.aisle = cid
        self.shelves = [2*cid-1, 2*cid]
        self.busy_until = 0.0
        self.position = 'origin'  # 'origin' or (shelf,z,x,slot)
        self.last_op = None       # 'inbound'/'outbound'/None
        self.carrying = None      # box dict being carried (inbound)
        self.outbound_q = []      # pending outbound orders
        self.inbound_q = []       # pending inbound boxes
        self.timeline = []        # completed task records
        self.total_compound = 0   # compound operation count

    def is_idle(self, t):
        return self.busy_until <= t

# Initialize 7 cranes
cranes = {i: Crane(i) for i in range(1, NUM_CRANES + 1)}

# ============ Event queue ============
events = []  # (time, counter, type, payload) — counter for tie-breaking
ev_cnt = [0]  # mutable counter for global access

def push_event(time, etype, payload):
    """Push event with unique counter for tie-breaking"""
    ev_cnt[0] += 1
    heapq.heappush(events, (time, ev_cnt[0], etype, payload))

# Schedule all outbound arrivals
for order in outbound_list:
    push_event(order['arrival'], 'outbound_arrival', order)

# Schedule all inbound arrivals
for batch in inbound_batches:
    push_event(batch['arrival'], 'inbound_arrival', batch)

# ============ Statistics ============
total_outbound_completed = 0
total_inbound_completed = 0
total_outbound_flow = 0.0
total_inbound_flow = 0.0
total_pure_out = 0
total_pure_in = 0
total_compound_ops = 0
pending_outbound = []  # outbound orders waiting for stock replenishment

# ============ Execution functions ============
def execute_pure_outbound(t, crane):
    """纯出库：原点→出库位→取货→回原点"""
    global total_outbound_completed, total_outbound_flow, total_pure_out

    # Find first executable outbound order within this crane's aisle
    idx = None
    for i, order in enumerate(crane.outbound_q):
        mid = order['material']
        if mid in mat_positions and len(mat_positions[mid]) > 0:
            # Select best position in this crane's aisle
            aisle_positions = [p for p in mat_positions[mid] if get_aisle(p['shelf']) == crane.aisle]
            if not aisle_positions:
                continue  # positions exist but not in this aisle — will be redistributed
            best_pos = min(aisle_positions, key=lambda p: p['T_out'])
            order['assigned_pos'] = best_pos
            idx = i
            break
    if idx is None:
        return False  # no executable order

    order = crane.outbound_q.pop(idx)
    pos = order['assigned_pos']
    t_out_val = pos['T_out']
    duration = 2 * t_out_val + T0
    flow_time = (t + duration) - order['arrival']

    crane.busy_until = t + duration
    crane.position = 'origin'
    crane.last_op = 'outbound'

    # Remove position from mat_positions (box picked)
    mat_positions[order['material']].remove(pos)
    # Free the position for future use
    shelf, z, x, slot = pos['shelf'], pos['z'], pos['x'], pos['slot']
    occupied[shelf-1][z][x][slot] = None
    pos_tuple = (shelf, z, x, slot)
    empty_set.add(pos_tuple)
    # Insert into empty_by_aisle in sorted order
    aisle = get_aisle(shelf)
    t_pos = T_out(x, z)
    ins_idx = 0
    for j, ep in enumerate(empty_by_aisle[aisle]):
        if T_out(ep[2], ep[1]) > t_pos:
            ins_idx = j; break
        ins_idx = j + 1
    empty_by_aisle[aisle].insert(ins_idx, pos_tuple)

    crane.timeline.append({
        'type': 'outbound', 'order_id': order['id'],
        'material': order['material'], 'box_type': order['box_type'],
        'arrival': order['arrival'], 'start': t, 'end': t + duration,
        'op_time': duration, 'flow_time': flow_time,
        'shelf': shelf, 'z': z, 'x': x, 'slot': slot,
        'compound': False
    })

    total_outbound_completed += 1
    total_outbound_flow += flow_time
    total_pure_out += 1
    return True

def execute_pure_inbound(t, crane):
    """纯入库：原点→入库位→放货（放到即完成，回程为系统开销）"""
    global total_inbound_completed, total_inbound_flow, total_pure_in

    if not crane.inbound_q:
        return False

    # SPT: shortest T_store first
    crane.inbound_q.sort(key=lambda b: b['T_store'])

    # Find nearest empty for first box (recompute since state may have changed)
    box = crane.inbound_q.pop(0)
    new_pos = find_nearest_empty_aisle(box['box_type'], crane.aisle)
    if new_pos is None:
        crane.inbound_q.insert(0, box)
        return False

    shelf, z, x, slot = new_pos
    t_out_val = T_out(x, z)

    # Inbound completes when box is placed at storage
    op_time = t_out_val + T0  # go + place (no return needed for completion)
    flow_time = (t + op_time) - box['arrival']

    # Crane is busy for full round trip (return to origin is system overhead)
    full_duration = 2 * t_out_val + T0

    crane.busy_until = t + full_duration
    crane.position = 'origin'  # returns to origin after pure inbound
    crane.last_op = 'inbound'

    # Occupy the position
    empty_set.discard(new_pos)
    occupied[shelf-1][z][x][slot] = box['material']
    mat_positions.setdefault(box['material'], []).append(
        {'shelf': shelf, 'z': z, 'x': x, 'slot': slot, 'T_out': t_out_val})

    crane.timeline.append({
        'type': 'inbound', 'batch_id': box['batch_id'],
        'material': box['material'], 'box_type': box['box_type'],
        'arrival': box['arrival'], 'start': t, 'end': t + op_time,
        'op_time': op_time, 'flow_time': flow_time,
        'shelf': shelf, 'z': z, 'x': x, 'slot': slot,
        'compound': False
    })

    total_inbound_completed += 1
    total_inbound_flow += flow_time
    total_pure_in += 1
    return True

def execute_compound(t, crane):
    """复合作业：出入库交替，消除空回程"""
    global total_outbound_completed, total_inbound_flow, total_outbound_flow
    global total_inbound_completed, total_compound_ops

    if not crane.outbound_q or not crane.inbound_q:
        return False

    # Sort queues (SPT for outbound, SPT for inbound)
    crane.outbound_q.sort(key=lambda o: (
        min((p['T_out'] for p in mat_positions.get(o['material'], [{'T_out': float('inf')}])),
            default=float('inf'))
    ))
    crane.inbound_q.sort(key=lambda b: b['T_store'])

    # Get best executable outbound within this crane's aisle
    out_idx = None; out_order = None; out_pos = None
    for i, o in enumerate(crane.outbound_q):
        if o['material'] in mat_positions and mat_positions[o['material']]:
            aisle_positions = [p for p in mat_positions[o['material']] if get_aisle(p['shelf']) == crane.aisle]
            if not aisle_positions:
                continue
            out_pos = min(aisle_positions, key=lambda p: p['T_out'])
            out_order = o; out_idx = i; break
    if out_idx is None:
        return False

    in_box = crane.inbound_q[0]
    in_pos = find_nearest_empty_aisle(in_box['box_type'], crane.aisle)
    if in_pos is None:
        return False

    in_shelf, in_z, in_x, in_slot = in_pos
    t_out_ob = out_pos['T_out']
    t_out_ib = T_out(in_x, in_z)
    t_between_val = T_between(in_x, in_z, out_pos['x'], out_pos['z'])

    at_origin = (crane.position == 'origin')

    if at_origin:
        # Sequence: origin→ib(T_out_ib+T0) → ob(T_between+T_out_ob+T0) → origin
        ib_op_time = t_out_ib + T0
        ob_op_time = t_between_val + t_out_ob + T0
        total_duration = ib_op_time + ob_op_time
        ib_start = t; ib_end = t + ib_op_time
        ob_start = ib_end; ob_end = ib_end + ob_op_time
        end_position = 'origin'; last_op = 'outbound'
    else:
        # Sequence: ib_prev→ob(T_between_prev+T_out_ob+T0)→origin
        #          →ib_next(T_out_ib+T0)
        _, ib_prev_z, ib_prev_x, _ = crane.position
        t_between_prev = T_between(ib_prev_x, ib_prev_z, out_pos['x'], out_pos['z'])
        ob_op_time = t_between_prev + t_out_ob + T0
        ib_op_time = t_out_ib + T0
        total_duration = ob_op_time + ib_op_time
        ob_start = t; ob_end = t + ob_op_time
        ib_start = ob_end; ib_end = ob_end + ib_op_time
        end_position = (in_shelf, in_z, in_x, in_slot); last_op = 'inbound'

    ob_flow = ob_end - out_order['arrival']
    ib_flow = ib_end - in_box['arrival']

    # Execute
    crane.outbound_q.pop(out_idx)
    crane.inbound_q.pop(0)
    crane.busy_until = t + total_duration
    crane.position = end_position
    crane.last_op = last_op

    # Warehouse state: free outbound position
    mat_positions[out_order['material']].remove(out_pos)
    ob_shelf, ob_z, ob_x, ob_sl = out_pos['shelf'], out_pos['z'], out_pos['x'], out_pos['slot']
    occupied[ob_shelf-1][ob_z][ob_x][ob_sl] = None
    ob_tuple = (ob_shelf, ob_z, ob_x, ob_sl)
    empty_set.add(ob_tuple)
    ob_aisle = get_aisle(ob_shelf)
    t_ob_pos = T_out(ob_x, ob_z)
    ins_idx = 0
    for j, ep in enumerate(empty_by_aisle[ob_aisle]):
        if T_out(ep[2], ep[1]) > t_ob_pos:
            ins_idx = j; break
        ins_idx = j + 1
    empty_by_aisle[ob_aisle].insert(ins_idx, ob_tuple)

    # Warehouse state: occupy inbound position
    empty_set.discard(in_pos)
    occupied[in_shelf-1][in_z][in_x][in_slot] = in_box['material']
    mat_positions.setdefault(in_box['material'], []).append(
        {'shelf': in_shelf, 'z': in_z, 'x': in_x, 'slot': in_slot, 'T_out': t_out_ib})

    # Record timeline
    crane.timeline.append({
        'type': 'outbound', 'order_id': out_order['id'],
        'material': out_order['material'], 'box_type': out_order['box_type'],
        'arrival': out_order['arrival'],
        'start': ob_start, 'end': ob_end,
        'op_time': ob_op_time, 'flow_time': ob_flow,
        'shelf': ob_shelf, 'z': ob_z, 'x': ob_x, 'slot': ob_sl,
        'compound': True
    })
    crane.timeline.append({
        'type': 'inbound', 'batch_id': in_box['batch_id'],
        'material': in_box['material'], 'box_type': in_box['box_type'],
        'arrival': in_box['arrival'],
        'start': ib_start, 'end': ib_end,
        'op_time': ib_op_time, 'flow_time': ib_flow,
        'shelf': in_shelf, 'z': in_z, 'x': in_x, 'slot': in_slot,
        'compound': True
    })

    total_outbound_completed += 1
    total_inbound_completed += 1
    total_outbound_flow += ob_flow
    total_inbound_flow += ib_flow
    total_compound_ops += 2
    return True

# ============ Event handlers ============
def handle_outbound_arrival(t, order):
    """出库订单到达：分配到对应巷道，无库存则加入全局等待队列"""
    mid = order['material']
    if mid in mat_positions and mat_positions[mid]:
        best_pos = min(mat_positions[mid], key=lambda p: p['T_out'])
        aisle = get_aisle(best_pos['shelf'])
        cranes[aisle].outbound_q.append(order)
        if cranes[aisle].is_idle(t):
            push_event(t, 'crane_free', aisle)
    else:
        # Material not in stock yet — wait in global pending queue
        pending_outbound.append(order)

def handle_inbound_arrival(t, batch):
    """入库批次到达：展开为单箱，每箱找最近空位，按SPT入队"""
    boxes_created = 0
    for i in range(batch['box_count']):
        pos = find_nearest_empty_global(batch['box_type'])
        if pos is None:
            print(f"  WARNING: No empty position for {batch['box_type']} box {i+1}/{batch['box_count']}")
            continue
        shelf, z, x, slot = pos
        T_store = T_out(x, z) + T0
        aisle = get_aisle(shelf)

        box = {
            'batch_id': batch['batch_id'], 'material': batch['material'],
            'box_type': batch['box_type'], 'arrival': batch['arrival'],
            'shelf': shelf, 'z': z, 'x': x, 'slot': slot,
            'T_store': T_store, 'aisle': aisle
        }
        cranes[aisle].inbound_q.append(box)
        boxes_created += 1

    # Sort each affected crane's inbound queue by SPT
    affected_aisles = set()
    for i in range(min(batch['box_count'], boxes_created)):
        # Re-get the boxes we just added to find their aisles
        pass
    # Simpler: re-sort all affected cranes
    for a in range(1, NUM_CRANES + 1):
        if cranes[a].inbound_q:
            cranes[a].inbound_q.sort(key=lambda b: b['T_store'])
        if cranes[a].is_idle(t):
            push_event(t, 'crane_free', a)

def handle_crane_free(t, crane_id):
    """堆垛机空闲：决策下一任务"""
    crane = cranes[crane_id]

    # Guard: skip if crane is actually busy (duplicate crane_free events at same t)
    if not crane.is_idle(t):
        return

    # Handle stuck at non-origin (compound break)
    if crane.position != 'origin':
        pos = crane.position
        t_return = T_out(pos[2], pos[1])  # return trip time
        crane.busy_until = t + t_return
        crane.position = 'origin'
        crane.timeline.append({
            'type': 'empty_return', 'start': t, 'end': t + t_return,
            'op_time': t_return, 'flow_time': 0,
            'shelf': pos[0], 'z': pos[1], 'x': pos[2], 'slot': pos[3],
            'compound': False
        })
        push_event(t + t_return, 'crane_free', crane_id)
        return

    # Check global pending outbound: any orders whose materials are now in stock?
    still_pending = []
    for order in pending_outbound:
        mid = order['material']
        if mid in mat_positions and mat_positions[mid]:
            best_pos = min(mat_positions[mid], key=lambda p: p['T_out'])
            aisle = get_aisle(best_pos['shelf'])
            cranes[aisle].outbound_q.append(order)
            if cranes[aisle].is_idle(t):
                push_event(t, 'crane_free', aisle)
        else:
            still_pending.append(order)
    pending_outbound[:] = still_pending

    # Redistribute outbound orders if material positions changed (new inbound stock)
    new_outbound_q = []
    for order in crane.outbound_q:
        mid = order['material']
        if mid in mat_positions and mat_positions[mid]:
            best_pos = min(mat_positions[mid], key=lambda p: p['T_out'])
            correct_aisle = get_aisle(best_pos['shelf'])
            if correct_aisle != crane.aisle:
                cranes[correct_aisle].outbound_q.append(order)
                if cranes[correct_aisle].is_idle(t):
                    push_event(t, 'crane_free', correct_aisle)
            else:
                new_outbound_q.append(order)
        else:
            # Stock depleted — move back to pending
            pending_outbound.append(order)
    crane.outbound_q = new_outbound_q

    # Sort queues
    if crane.outbound_q:
        # Sort by best-possible T_total (SPT)
        def ob_sort_key(o):
            if o['material'] in mat_positions and mat_positions[o['material']]:
                best_t = min(p['T_out'] for p in mat_positions[o['material']])
                return 2 * best_t + T0
            return float('inf')
        crane.outbound_q.sort(key=ob_sort_key)
    if crane.inbound_q:
        crane.inbound_q.sort(key=lambda b: b['T_store'])

    has_ob = bool(crane.outbound_q)
    has_ib = bool(crane.inbound_q)

    if not has_ob and not has_ib:
        return  # idle

    # Check if top outbound is actually executable
    ob_executable = False
    if has_ob:
        top_ob = crane.outbound_q[0]
        if top_ob['material'] in mat_positions and mat_positions[top_ob['material']]:
            ob_executable = True

    if ob_executable and has_ib:
        # Compound mode
        if not execute_compound(t, crane):
            # Fallback: try pure outbound
            if not execute_pure_outbound(t, crane):
                execute_pure_inbound(t, crane)
    elif ob_executable:
        execute_pure_outbound(t, crane)
    elif has_ib:
        execute_pure_inbound(t, crane)
    else:
        # Neither executable (wait for inbound replenishment)
        # Don't push crane_free — will be woken by future inbound_arrival
        return

    push_event(crane.busy_until, 'crane_free', crane_id)

# ============ Main simulation loop ============
print("\n开始仿真...")
sim_events_processed = 0

while events:
    t, _, etype, payload = heapq.heappop(events)

    if etype == 'outbound_arrival':
        handle_outbound_arrival(t, payload)
    elif etype == 'inbound_arrival':
        handle_inbound_arrival(t, payload)
    elif etype == 'crane_free':
        handle_crane_free(t, payload)

    sim_events_processed += 1
    if sim_events_processed % 2000 == 0:
        remaining = len(events)
        print(f"  已处理{sim_events_processed}事件, 剩余{remaining}, t={t/3600:.1f}h")

print(f"仿真完成! 处理{sim_events_processed}事件")

# Process any remaining crane_free until all cranes idle
# (handled by the loop continuing until events exhausted)

# ============ Results summary ============
print("\n" + "=" * 60)
print("调度结果统计")
print("=" * 60)

print(f"出库完成: {total_outbound_completed}/{len(outbound_list)}")
print(f"入库完成: {total_inbound_completed}/{total_inbound_boxes}")
print(f"总流动时间: {total_outbound_flow + total_inbound_flow:.0f}s")
print(f"  出库总流动: {total_outbound_flow:.0f}s, 平均: {total_outbound_flow/total_outbound_completed:.0f}s" if total_outbound_completed > 0 else "  出库: 0")
print(f"  入库总流动: {total_inbound_flow:.0f}s, 平均: {total_inbound_flow/total_outbound_completed:.0f}s" if total_inbound_completed > 0 else "  入库: 0")
avg_out_flow = total_outbound_flow / total_outbound_completed if total_outbound_completed > 0 else 0
avg_in_flow = total_inbound_flow / total_inbound_completed if total_inbound_completed > 0 else 0
avg_all_flow = (total_outbound_flow + total_inbound_flow) / (total_outbound_completed + total_inbound_completed) if (total_outbound_completed + total_inbound_completed) > 0 else 0
print(f"平均流动时间: 出库{avg_out_flow:.0f}s, 入库{avg_in_flow:.0f}s, 总体{avg_all_flow:.0f}s")

print(f"\n操作统计:")
print(f"  纯出库: {total_pure_out}, 纯入库: {total_pure_in}, 复合: {total_compound_ops}")
total_ops = total_pure_out + total_pure_in + total_compound_ops
if total_ops > 0:
    print(f"  复合占比: {total_compound_ops/total_ops*100:.1f}%")

print(f"\n各堆垛机负载:")
for a in range(1, NUM_CRANES + 1):
    c = cranes[a]
    n_tasks = len([e for e in c.timeline if e['type'] != 'empty_return'])
    ob_tasks = len([e for e in c.timeline if e['type'] == 'outbound'])
    ib_tasks = len([e for e in c.timeline if e['type'] == 'inbound'])
    empty_returns = len([e for e in c.timeline if e['type'] == 'empty_return'])
    makespan = max([e['end'] for e in c.timeline]) if c.timeline else 0
    print(f"  堆垛机{a}(巷道{a}): {n_tasks}任务(出{ob_tasks}/入{ib_tasks}), "
          f"空回{empty_returns}次, 完工{makespan/3600:.1f}h")

# ============ Output CSVs ============
print(f"\n保存CSV至 {CSV_DIR}...")

# 1. Crane timeline
timeline_rows = []
for a in range(1, NUM_CRANES + 1):
    for entry in cranes[a].timeline:
        if entry['type'] == 'empty_return':
            continue  # skip empty returns in main output (or include with note)
        timeline_rows.append({
            '堆垛机编号': a, '巷道号': a,
            '任务类型': entry['type'],
            '订单号': entry.get('order_id', entry.get('batch_id', '')),
            '原材料编号': entry.get('material', ''),
            '箱子类型': entry.get('box_type', ''),
            '到达时刻(s)': entry.get('arrival', 0),
            '开始时刻(s)': entry['start'],
            '完成时刻(s)': entry['end'],
            '操作时间(s)': entry['op_time'],
            '等待时间(s)': entry['start'] - entry.get('arrival', entry['start']),
            '流动时间(s)': entry['flow_time'],
            '货架号': entry['shelf'],
            '层(z)': entry['z'],
            '列(x)': entry['x'],
            '深浅位': '浅位' if entry['slot'] == 0 else '深位',
            '是否复合': '是' if entry['compound'] else '否'
        })
tl_df = pd.DataFrame(timeline_rows)
tl_df.to_csv(os.path.join(CSV_DIR, '各堆垛机作业时间表.csv'), index=False, encoding='utf-8-sig')
print(f"  作业时间表: {len(tl_df)}条记录")

# 2. Inbound allocation
ib_rows = []
for a in range(1, NUM_CRANES + 1):
    for entry in cranes[a].timeline:
        if entry['type'] == 'inbound':
            ib_rows.append({
                '入库批次': entry.get('batch_id', ''),
                '原材料编号': entry.get('material', ''),
                '箱子类型': entry.get('box_type', ''),
                '堆垛机编号': a,
                '货架号': entry['shelf'],
                '层(z)': entry['z'],
                '列(x)': entry['x'],
                '深浅位': '浅位' if entry['slot'] == 0 else '深位',
                '到达时刻(s)': entry.get('arrival', 0),
                '开始时刻(s)': entry['start'],
                '完成时刻(s)': entry['end'],
                '流动时间(s)': entry['flow_time'],
                '是否复合': '是' if entry['compound'] else '否'
            })
ib_df = pd.DataFrame(ib_rows)
ib_df.to_csv(os.path.join(CSV_DIR, '入库货位分配方案.csv'), index=False, encoding='utf-8-sig')
print(f"  入库分配: {len(ib_df)}条记录")

# 3. Outbound sequence
ob_rows = []
for a in range(1, NUM_CRANES + 1):
    for entry in cranes[a].timeline:
        if entry['type'] == 'outbound':
            ob_rows.append({
                '订单号': entry.get('order_id', ''),
                '原材料编号': entry.get('material', ''),
                '箱子类型': entry.get('box_type', ''),
                '堆垛机编号': a,
                '货架号': entry['shelf'],
                '层(z)': entry['z'],
                '列(x)': entry['x'],
                '深浅位': '浅位' if entry['slot'] == 0 else '深位',
                '到达时刻(s)': entry.get('arrival', 0),
                '开始时刻(s)': entry['start'],
                '完成时刻(s)': entry['end'],
                '操作时间(s)': entry['op_time'],
                '等待时间(s)': entry['start'] - entry.get('arrival', entry['start']),
                '流动时间(s)': entry['flow_time'],
                '是否复合': '是' if entry['compound'] else '否'
            })
ob_df = pd.DataFrame(ob_rows)
ob_df.to_csv(os.path.join(CSV_DIR, '出库作业序列.csv'), index=False, encoding='utf-8-sig')
print(f"  出库序列: {len(ob_df)}条记录")

# ============ Self-checks ============
print("\n" + "=" * 60)
print("自检")
print("=" * 60)

# Data integrity
print(f"[数据] 入库箱数: {total_inbound_completed} (预期{total_inbound_boxes}): "
      f"{'PASS' if total_inbound_completed == total_inbound_boxes else 'FAIL'}")
print(f"[数据] 出库单数: {total_outbound_completed} (预期{len(outbound_list)}): "
      f"{'PASS' if total_outbound_completed == len(outbound_list) else 'FAIL'}")

# Layer constraints
e4_violations = sum(1 for r in ib_rows if r['箱子类型'] == 'E4' and r['层(z)'] < 43)
e3_violations = sum(1 for r in ib_rows if r['箱子类型'] == 'E3' and r['层(z)'] < 9)
print(f"[层数] E4≥43层违规: {e4_violations} {'PASS' if e4_violations==0 else 'FAIL'}")
print(f"[层数] E3≥9层违规: {e3_violations} {'PASS' if e3_violations==0 else 'FAIL'}")

# No temporal overlap on inbound slots (same slot used at different times is OK)
slot_usage = {}  # (shelf,z,x,slot) -> [(start,end), ...]
for a in range(1, NUM_CRANES + 1):
    for entry in cranes[a].timeline:
        if entry['type'] == 'inbound':
            key = (entry['shelf'], entry['z'], entry['x'], entry['slot'])
            slot_usage.setdefault(key, []).append((entry['start'], entry['end']))
temporal_conflicts = 0
for key, intervals in slot_usage.items():
    intervals.sort()
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i-1][1]:  # overlap
            temporal_conflicts += 1
print(f"[空位] 入库slot时间重叠冲突: {temporal_conflicts} "
      f"{'PASS' if temporal_conflicts==0 else 'FAIL'}")
print(f"  注: 同slot不同时间复用不计入冲突")

# Flow time non-negative
neg_flow = sum(1 for r in timeline_rows if r['流动时间(s)'] < 0)
print(f"[时间] 负流动时间: {neg_flow} {'PASS' if neg_flow==0 else 'FAIL'}")

# Occupancy count
final_occupied = sum(1 for shelf in range(NUM_SHELVES) for z in range(1, NUM_LAYERS+1)
                     for x in range(1, NUM_COLUMNS+1) for slot in [0,1]
                     if occupied[shelf][z][x][slot] is not None)
expected_final = alloc_count - total_outbound_completed + total_inbound_completed
print(f"[库位] 最终占用: {final_occupied} (预期{expected_final}): "
      f"{'PASS' if final_occupied == expected_final else 'FAIL'}")

# Crane operation range check
print(f"\n各堆垛机管辖区检查:")
for a in range(1, NUM_CRANES + 1):
    c = cranes[a]
    violations = 0
    for entry in c.timeline:
        if entry['type'] in ['outbound', 'inbound']:
            if get_aisle(entry['shelf']) != a:
                violations += 1
    print(f"  堆垛机{a}(巷道{a}/货架{2*a-1}-{2*a}): 跨界违规{violations} "
          f"{'PASS' if violations==0 else 'FAIL'}")

print(f"\n目标函数: 总流动时间 = {total_outbound_flow + total_inbound_flow:.0f}s")
print(f"  = {total_outbound_flow:.0f}(出库) + {total_inbound_flow:.0f}(入库)")
print(f"\nDone! 文件保存于 {CSV_DIR}")
