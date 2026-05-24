"""
问题四：出入库协同调度优化（参数化时间模型版本）
- 支持线性模型 (LinearTimeModel) 和非线性模型 (NonlinearTimeModel)
- 非线性模型区分空载/满载，含梯形/三角形速度曲线
- 其余调度逻辑（事件驱动、SPT、复合作业）与问题三一致

操作时间公式（非线性）:
  纯出库: T_out(empty) + T0 + T_out(loaded)
  纯入库: T_out(loaded) + T0 (放到即完成), busy含空回T_out(empty)
  复合(at origin):   ib段(loaded) + ob段(empty→loaded)
  复合(not origin):  ob段(prev→ob,empty→loaded) + ib段(loaded)
"""
import pandas as pd, numpy as np, os, heapq, sys

# Add parent for import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from 问题四_时间模型 import (LinearTimeModel, NonlinearTimeModel, COL_WIDTH, LH)


def get_aisle(shelf):
    return (shelf - 1) // 2 + 1


def layer_allowed(z):
    if z <= 8: return ['E1']
    elif z <= 42: return ['E1', 'E3']
    return ['E1', 'E3', 'E4']


def run_simulation(model, output_csv_dir):
    """主仿真，返回统计字典"""
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    NUM_SHELVES, NUM_LAYERS, NUM_COLUMNS = 14, 50, 68
    NUM_CRANES = 7

    # ============ Data loading ============
    print("=" * 60)
    print(f"问题四 调度仿真: {model.name}")
    print(f"T0={model.T0}s, COL_WIDTH={model.COL_WIDTH}m, {NUM_CRANES}台堆垛机")
    print("=" * 60)

    # 1. Problem 1 allocation
    alloc_df = pd.read_csv(os.path.join(BASE, '输出结果', '问题一', '分配结果CSV', '完整分配明细.csv'))
    inv_df = pd.read_csv(os.path.join(BASE, '清洗数据一', '清洗后_原材料库存数据.csv'))

    # 2. Outbound orders
    out_orders = pd.read_csv(os.path.join(BASE, '清洗数据一', '清洗后_生产线订料数据.csv'))
    out_orders['时间'] = pd.to_datetime(out_orders['时间'])
    out_orders = out_orders.sort_values('时间').reset_index(drop=True)

    # 3. Inbound orders
    in_orders = pd.read_csv(os.path.join(BASE, '清洗数据一', '清洗后_入库材料数据.csv'))
    in_orders['入库时间'] = pd.to_datetime(in_orders['入库时间'])
    in_orders = in_orders.sort_values('入库时间').reset_index(drop=True)

    t0_ref = min(out_orders['时间'].min(), in_orders['入库时间'].min())
    print(f"t0_ref = {t0_ref}")

    # ============ Warehouse state ============
    occupied = [[[[None for _ in range(2)] for _ in range(NUM_COLUMNS + 1)]
                 for _ in range(NUM_LAYERS + 1)] for _ in range(NUM_SHELVES)]

    mat_positions = {}
    slot_map = {'浅位': 0, '深位': 1}

    # T_out for initial sort key (use loaded state for inbound-centric empty_by_aisle)
    def sort_t_out(x, z):
        """Sort key for empty positions: use loaded T_out for inbound estimate"""
        return model.T_out(x, z, 'loaded')

    for _, r in alloc_df.iterrows():
        shelf = int(r['货架(y)'])
        z = int(r['层(z)'])
        x = int(r['列(x)'])
        slot = slot_map[r['深浅位']]
        mid = r['原材料编号']
        occupied[shelf-1][z][x][slot] = mid
        mat_positions.setdefault(mid, []).append({
            'shelf': shelf, 'z': z, 'x': x, 'slot': slot,
            'T_out': model.T_out(x, z, 'empty')  # for outbound: empty go estimate
        })

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

    for a in range(1, NUM_CRANES + 1):
        empty_by_aisle[a].sort(key=lambda p: sort_t_out(p[2], p[1]))
    print(f"初始空位: {len(empty_set)}")

    # ============ Prepare orders ============
    outbound_list = []
    for _, r in out_orders.iterrows():
        mid = r['原材料号']
        arrival = (r['时间'] - t0_ref).total_seconds()
        outbound_list.append({
            'id': r['订单号'], 'material': mid,
            'box_type': r['箱子类型'], 'arrival': arrival,
            'assigned_pos': None
        })
    print(f"出库订单: {len(outbound_list)}单")

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

    # ============ Empty position search ============
    def find_nearest_empty_global(box_type):
        best = None; best_t = float('inf')
        for a in range(1, NUM_CRANES + 1):
            for pos in empty_by_aisle[a]:
                shelf, z, x, slot = pos
                if pos not in empty_set: continue
                if box_type not in layer_allowed(z): continue
                t = model.T_out(x, z, 'loaded')
                if t < best_t:
                    best_t = t; best = pos
                break
        return best

    def find_nearest_empty_aisle(box_type, aisle):
        best = None; best_t = float('inf')
        for pos in empty_by_aisle[aisle]:
            shelf, z, x, slot = pos
            if pos not in empty_set: continue
            if box_type not in layer_allowed(z): continue
            t = model.T_out(x, z, 'loaded')
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
            self.position = 'origin'
            self.last_op = None
            self.carrying = None
            self.outbound_q = []
            self.inbound_q = []
            self.timeline = []
            self.total_compound = 0

        def is_idle(self, t):
            return self.busy_until <= t

    cranes = {i: Crane(i) for i in range(1, NUM_CRANES + 1)}

    # ============ Event queue ============
    import heapq
    events = []
    ev_cnt = [0]

    def push_event(time, etype, payload):
        ev_cnt[0] += 1
        heapq.heappush(events, (time, ev_cnt[0], etype, payload))

    for order in outbound_list:
        push_event(order['arrival'], 'outbound_arrival', order)
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
    pending_outbound = []

    def insert_empty_sorted(pos_tuple):
        """Insert freed position into empty_by_aisle in sorted order"""
        shelf, z, x, slot = pos_tuple
        aisle = get_aisle(shelf)
        t_pos = model.T_out(x, z, 'loaded')
        ins_idx = 0
        for j, ep in enumerate(empty_by_aisle[aisle]):
            if model.T_out(ep[2], ep[1], 'loaded') > t_pos:
                ins_idx = j; break
            ins_idx = j + 1
        empty_by_aisle[aisle].insert(ins_idx, pos_tuple)

    # ============ Execution functions ============
    def execute_pure_outbound(t, crane):
        nonlocal total_outbound_completed, total_outbound_flow, total_pure_out

        idx = None
        for i, order in enumerate(crane.outbound_q):
            mid = order['material']
            if mid in mat_positions and len(mat_positions[mid]) > 0:
                aisle_positions = [p for p in mat_positions[mid] if get_aisle(p['shelf']) == crane.aisle]
                if not aisle_positions: continue
                best_pos = min(aisle_positions, key=lambda p: p['T_out'])
                order['assigned_pos'] = best_pos
                idx = i; break
        if idx is None:
            return False

        order = crane.outbound_q.pop(idx)
        pos = order['assigned_pos']
        x, z = pos['x'], pos['z']

        # Nonlinear: go empty, return loaded
        op_time = model.T_out(x, z, 'empty') + model.T0 + model.T_out(x, z, 'loaded')
        flow_time = (t + op_time) - order['arrival']

        crane.busy_until = t + op_time
        crane.position = 'origin'
        crane.last_op = 'outbound'

        mat_positions[order['material']].remove(pos)
        shelf, z, x, slot = pos['shelf'], pos['z'], pos['x'], pos['slot']
        occupied[shelf-1][z][x][slot] = None
        pos_tuple = (shelf, z, x, slot)
        empty_set.add(pos_tuple)
        insert_empty_sorted(pos_tuple)

        crane.timeline.append({
            'type': 'outbound', 'order_id': order['id'],
            'material': order['material'], 'box_type': order['box_type'],
            'arrival': order['arrival'], 'start': t, 'end': t + op_time,
            'op_time': op_time, 'flow_time': flow_time,
            'shelf': shelf, 'z': z, 'x': x, 'slot': slot,
            'compound': False
        })

        total_outbound_completed += 1
        total_outbound_flow += flow_time
        total_pure_out += 1
        return True

    def execute_pure_inbound(t, crane):
        nonlocal total_inbound_completed, total_inbound_flow, total_pure_in

        if not crane.inbound_q:
            return False

        crane.inbound_q.sort(key=lambda b: b['T_store'])

        box = crane.inbound_q.pop(0)
        new_pos = find_nearest_empty_aisle(box['box_type'], crane.aisle)
        if new_pos is None:
            crane.inbound_q.insert(0, box)
            return False

        shelf, z, x, slot = new_pos

        # Nonlinear: go loaded (carrying box), place, return empty (overhead)
        op_time = model.T_out(x, z, 'loaded') + model.T0
        flow_time = (t + op_time) - box['arrival']

        # Crane busy for full round trip
        full_duration = model.T_out(x, z, 'loaded') + model.T0 + model.T_out(x, z, 'empty')

        crane.busy_until = t + full_duration
        crane.position = 'origin'
        crane.last_op = 'inbound'

        empty_set.discard(new_pos)
        occupied[shelf-1][z][x][slot] = box['material']
        mat_positions.setdefault(box['material'], []).append({
            'shelf': shelf, 'z': z, 'x': x, 'slot': slot,
            'T_out': model.T_out(x, z, 'empty')
        })

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
        nonlocal total_outbound_completed, total_inbound_flow, total_outbound_flow
        nonlocal total_inbound_completed, total_compound_ops

        if not crane.outbound_q or not crane.inbound_q:
            return False

        # Sort queues (SPT for both)
        def ob_sort_key(o):
            if o['material'] in mat_positions and mat_positions[o['material']]:
                p = min(mat_positions[o['material']], key=lambda x: x['T_out'])
                return model.T_out(p['x'], p['z'], 'empty') + model.T0 + model.T_out(p['x'], p['z'], 'loaded')
            return float('inf')
        crane.outbound_q.sort(key=ob_sort_key)
        crane.inbound_q.sort(key=lambda b: b['T_store'])

        out_idx = None; out_order = None; out_pos = None
        for i, o in enumerate(crane.outbound_q):
            if o['material'] in mat_positions and mat_positions[o['material']]:
                aisle_positions = [p for p in mat_positions[o['material']] if get_aisle(p['shelf']) == crane.aisle]
                if not aisle_positions: continue
                out_pos = min(aisle_positions, key=lambda p: p['T_out'])
                out_order = o; out_idx = i; break
        if out_idx is None:
            return False

        in_box = crane.inbound_q[0]
        in_pos = find_nearest_empty_aisle(in_box['box_type'], crane.aisle)
        if in_pos is None:
            return False

        in_shelf, in_z, in_x, in_slot = in_pos
        ob_x, ob_z = out_pos['x'], out_pos['z']

        at_origin = (crane.position == 'origin')

        if at_origin:
            # origin→ib(loaded)→place → ob(empty→loaded)→origin
            ib_op_time = model.T_out(in_x, in_z, 'loaded') + model.T0
            ob_op_time = (model.T_between(in_x, in_z, ob_x, ob_z, 'empty')
                          + model.T0
                          + model.T_out(ob_x, ob_z, 'loaded'))
            total_duration = ib_op_time + ob_op_time
            ib_start = t; ib_end = t + ib_op_time
            ob_start = ib_end; ob_end = ib_end + ob_op_time
            end_position = 'origin'; last_op = 'outbound'
        else:
            # ib_prev→ob(empty→loaded)→origin → origin→ib(loaded)→place
            _, ib_prev_z, ib_prev_x, _ = crane.position
            ob_op_time = (model.T_between(ib_prev_x, ib_prev_z, ob_x, ob_z, 'empty')
                          + model.T0
                          + model.T_out(ob_x, ob_z, 'loaded'))
            ib_op_time = model.T_out(in_x, in_z, 'loaded') + model.T0
            total_duration = ob_op_time + ib_op_time
            ob_start = t; ob_end = t + ob_op_time
            ib_start = ob_end; ib_end = ob_end + ib_op_time
            end_position = (in_shelf, in_z, in_x, in_slot); last_op = 'inbound'

        ob_flow = ob_end - out_order['arrival']
        ib_flow = ib_end - in_box['arrival']

        crane.outbound_q.pop(out_idx)
        crane.inbound_q.pop(0)
        crane.busy_until = t + total_duration
        crane.position = end_position
        crane.last_op = last_op

        # Free outbound position
        mat_positions[out_order['material']].remove(out_pos)
        ob_shelf, ob_z, ob_x, ob_sl = out_pos['shelf'], out_pos['z'], out_pos['x'], out_pos['slot']
        occupied[ob_shelf-1][ob_z][ob_x][ob_sl] = None
        ob_tuple = (ob_shelf, ob_z, ob_x, ob_sl)
        empty_set.add(ob_tuple)
        insert_empty_sorted(ob_tuple)

        # Occupy inbound position
        empty_set.discard(in_pos)
        occupied[in_shelf-1][in_z][in_x][in_slot] = in_box['material']
        mat_positions.setdefault(in_box['material'], []).append({
            'shelf': in_shelf, 'z': in_z, 'x': in_x, 'slot': in_slot,
            'T_out': model.T_out(in_x, in_z, 'empty')
        })

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
        mid = order['material']
        if mid in mat_positions and mat_positions[mid]:
            best_pos = min(mat_positions[mid], key=lambda p: p['T_out'])
            aisle = get_aisle(best_pos['shelf'])
            cranes[aisle].outbound_q.append(order)
            if cranes[aisle].is_idle(t):
                push_event(t, 'crane_free', aisle)
        else:
            pending_outbound.append(order)

    def handle_inbound_arrival(t, batch):
        boxes_created = 0
        for i in range(batch['box_count']):
            pos = find_nearest_empty_global(batch['box_type'])
            if pos is None:
                print(f"  WARNING: No empty position for {batch['box_type']}")
                continue
            shelf, z, x, slot = pos
            T_store = model.T_out(x, z, 'loaded') + model.T0
            aisle = get_aisle(shelf)
            box = {
                'batch_id': batch['batch_id'], 'material': batch['material'],
                'box_type': batch['box_type'], 'arrival': batch['arrival'],
                'shelf': shelf, 'z': z, 'x': x, 'slot': slot,
                'T_store': T_store, 'aisle': aisle
            }
            cranes[aisle].inbound_q.append(box)
            boxes_created += 1

        for a in range(1, NUM_CRANES + 1):
            if cranes[a].inbound_q:
                cranes[a].inbound_q.sort(key=lambda b: b['T_store'])
            if cranes[a].is_idle(t):
                push_event(t, 'crane_free', a)

    def handle_crane_free(t, crane_id):
        crane = cranes[crane_id]

        if not crane.is_idle(t):
            return

        # Stuck at non-origin (compound break) — empty return
        if crane.position != 'origin':
            pos = crane.position
            t_return = model.T_out(pos[2], pos[1], 'empty')
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

        # Check global pending outbound
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

        # Redistribute outbound orders
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
                pending_outbound.append(order)
        crane.outbound_q = new_outbound_q

        # Sort queues
        if crane.outbound_q:
            def ob_sort_key(o):
                if o['material'] in mat_positions and mat_positions[o['material']]:
                    p = min(mat_positions[o['material']], key=lambda x: x['T_out'])
                    return model.T_out(p['x'], p['z'], 'empty') + model.T0 + model.T_out(p['x'], p['z'], 'loaded')
                return float('inf')
            crane.outbound_q.sort(key=ob_sort_key)
        if crane.inbound_q:
            crane.inbound_q.sort(key=lambda b: b['T_store'])

        has_ob = bool(crane.outbound_q)
        has_ib = bool(crane.inbound_q)

        if not has_ob and not has_ib:
            return

        ob_executable = False
        if has_ob:
            top_ob = crane.outbound_q[0]
            if top_ob['material'] in mat_positions and mat_positions[top_ob['material']]:
                ob_executable = True

        if ob_executable and has_ib:
            if not execute_compound(t, crane):
                if not execute_pure_outbound(t, crane):
                    execute_pure_inbound(t, crane)
        elif ob_executable:
            execute_pure_outbound(t, crane)
        elif has_ib:
            execute_pure_inbound(t, crane)
        else:
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

    # ============ Results summary ============
    print("\n" + "=" * 60)
    print("调度结果统计")
    print("=" * 60)

    print(f"出库完成: {total_outbound_completed}/{len(outbound_list)}")
    print(f"入库完成: {total_inbound_completed}/{total_inbound_boxes}")

    avg_out_flow = total_outbound_flow / total_outbound_completed if total_outbound_completed > 0 else 0
    avg_in_flow = total_inbound_flow / total_inbound_completed if total_inbound_completed > 0 else 0
    total_flow = total_outbound_flow + total_inbound_flow
    total_all = total_outbound_completed + total_inbound_completed
    avg_all_flow = total_flow / total_all if total_all > 0 else 0

    print(f"总流动时间: {total_flow:.0f}s = {total_outbound_flow:.0f}(出库) + {total_inbound_flow:.0f}(入库)")
    print(f"平均流动时间: 出库{avg_out_flow:.1f}s, 入库{avg_in_flow:.1f}s, 总体{avg_all_flow:.1f}s")

    total_ops = total_pure_out + total_pure_in + total_compound_ops
    print(f"\n操作统计:")
    print(f"  纯出库: {total_pure_out}, 纯入库: {total_pure_in}, 复合: {total_compound_ops}")
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
    os.makedirs(output_csv_dir, exist_ok=True)
    print(f"\n保存CSV至 {output_csv_dir}...")

    # 1. Crane timeline
    timeline_rows = []
    for a in range(1, NUM_CRANES + 1):
        for entry in cranes[a].timeline:
            if entry['type'] == 'empty_return':
                continue
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
    tl_df.to_csv(os.path.join(output_csv_dir, '各堆垛机作业时间表.csv'), index=False, encoding='utf-8-sig')
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
    ib_df.to_csv(os.path.join(output_csv_dir, '入库货位分配方案.csv'), index=False, encoding='utf-8-sig')
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
    ob_df.to_csv(os.path.join(output_csv_dir, '出库作业序列.csv'), index=False, encoding='utf-8-sig')
    print(f"  出库序列: {len(ob_df)}条记录")

    # ============ Self-checks ============
    print("\n" + "=" * 60)
    print("自检")
    print("=" * 60)

    print(f"[数据] 入库箱数: {total_inbound_completed} (预期{total_inbound_boxes}): "
          f"{'PASS' if total_inbound_completed == total_inbound_boxes else 'FAIL'}")
    print(f"[数据] 出库单数: {total_outbound_completed} (预期{len(outbound_list)}): "
          f"{'PASS' if total_outbound_completed == len(outbound_list) else 'FAIL'}")

    e4_violations = sum(1 for r in ib_rows if r['箱子类型'] == 'E4' and r['层(z)'] < 43)
    e3_violations = sum(1 for r in ib_rows if r['箱子类型'] == 'E3' and r['层(z)'] < 9)
    print(f"[层数] E4≥43层违规: {e4_violations} {'PASS' if e4_violations==0 else 'FAIL'}")
    print(f"[层数] E3≥9层违规: {e3_violations} {'PASS' if e3_violations==0 else 'FAIL'}")

    # Temporal conflict check
    slot_usage = {}
    for a in range(1, NUM_CRANES + 1):
        for entry in cranes[a].timeline:
            if entry['type'] == 'inbound':
                key = (entry['shelf'], entry['z'], entry['x'], entry['slot'])
                slot_usage.setdefault(key, []).append((entry['start'], entry['end']))
    temporal_conflicts = 0
    for key, intervals in slot_usage.items():
        intervals.sort()
        for i in range(1, len(intervals)):
            if intervals[i][0] < intervals[i-1][1]:
                temporal_conflicts += 1
    print(f"[空位] 入库slot时间重叠冲突: {temporal_conflicts} "
          f"{'PASS' if temporal_conflicts==0 else 'FAIL'}")

    neg_flow = sum(1 for r in timeline_rows if r['流动时间(s)'] < 0)
    print(f"[时间] 负流动时间: {neg_flow} {'PASS' if neg_flow==0 else 'FAIL'}")

    final_occupied = sum(1 for shelf in range(NUM_SHELVES) for z in range(1, NUM_LAYERS+1)
                         for x in range(1, NUM_COLUMNS+1) for slot in [0,1]
                         if occupied[shelf][z][x][slot] is not None)
    expected_final = alloc_count - total_outbound_completed + total_inbound_completed
    print(f"[库位] 最终占用: {final_occupied} (预期{expected_final}): "
          f"{'PASS' if final_occupied == expected_final else 'FAIL'}")

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

    print(f"\n目标函数: 总流动时间 = {total_flow:.0f}s")

    # Return stats dict
    return {
        'model_name': model.name,
        'total_flow': total_flow,
        'outbound_flow': total_outbound_flow,
        'inbound_flow': total_inbound_flow,
        'avg_out_flow': avg_out_flow,
        'avg_in_flow': avg_in_flow,
        'avg_all_flow': avg_all_flow,
        'outbound_count': total_outbound_completed,
        'inbound_count': total_inbound_completed,
        'pure_out': total_pure_out,
        'pure_in': total_pure_in,
        'compound_ops': total_compound_ops,
        'total_ops': total_ops,
        'compound_ratio': total_compound_ops/total_ops*100 if total_ops > 0 else 0,
        'makespan': max(max([e['end'] for e in cranes[a].timeline]) if cranes[a].timeline else 0
                        for a in range(1, NUM_CRANES+1)),
        'pass_count': sum([
            total_inbound_completed == total_inbound_boxes,
            total_outbound_completed == len(outbound_list),
            e4_violations == 0,
            e3_violations == 0,
            temporal_conflicts == 0,
            neg_flow == 0,
            final_occupied == expected_final,
        ]),
        'total_checks': 7,
    }
