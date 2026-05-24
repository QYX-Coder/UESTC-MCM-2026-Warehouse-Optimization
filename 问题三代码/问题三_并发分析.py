"""7 crane concurrency analysis"""
import pandas as pd, numpy as np, os

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, '..', '输出结果', '问题三', '调度结果CSV', '各堆垛机作业时间表.csv')
tl = pd.read_csv(CSV)

def fmt(s):
    h = int(s//3600); m = int((s%3600)//60)
    return "{:.0f}h{:02d}m".format(h, m)

# Column name mapping (Chinese -> short)
cols = tl.columns.tolist()
c_crane = cols[0]   # 堆垛机编号
c_type = cols[2]    # 任务类型
c_order = cols[3]   # 订单号
c_arrive = cols[6]  # 到达时刻(s)
c_start = cols[7]   # 开始时刻(s)
c_end = cols[8]     # 完成时刻(s)
c_op = cols[9]      # 操作时间(s)
c_shelf = cols[13]  # 货架号
c_z = cols[14]      # 层(z)
c_x = cols[15]      # 列(x)

print('='*60)
print('Crane utilization')
print('='*60)
for cid in range(1, 8):
    ct = tl[tl[c_crane]==cid]
    busy = ct[c_op].sum()
    span = ct[c_end].max() - ct[c_start].min()
    util = busy / span * 100
    n = len(ct)
    print('  Crane{}: {} tasks, busy {:.1f}h, span {:.1f}h, util {:.1f}%'.format(
        cid, n, busy/3600, span/3600, util))

# Efficient concurrency: event-based
print('\n' + '='*60)
print('Concurrency distribution (time-weighted)')
print('='*60)

events = []
for _, row in tl.iterrows():
    events.append((row[c_start], +1))
    events.append((row[c_end], -1))
events.sort()

active = 0
conc_total = {}
prev_t = events[0][0]

for t, delta in events:
    dur = t - prev_t
    if dur > 0:
        conc_total[active] = conc_total.get(active, 0) + dur
    active += delta
    prev_t = t

total_dur = sum(conc_total.values())
for k in sorted(conc_total.keys()):
    pct = conc_total[k] / total_dur * 100
    bar = '#' * max(1, int(pct))
    print('  {} cranes: {:7.1f}h ({:5.1f}%) {}'.format(k, conc_total[k]/3600, pct, bar))

# Find 7-concurrent windows
print('\n' + '='*60)
print('Max concurrency windows')
print('='*60)

active2 = 0
prev_t2 = events[0][0]
windows = {i: [] for i in range(0, 8)}
in_window = {i: False for i in range(0, 8)}
win_start = {i: 0 for i in range(0, 8)}

for t, delta in events:
    # Check transitions
    for n in range(0, 8):
        if active2 == n and not in_window[n]:
            win_start[n] = prev_t2
            in_window[n] = True
        elif active2 != n and in_window[n]:
            windows[n].append((win_start[n], prev_t2))
            in_window[n] = False
    active2 += delta
    prev_t2 = t

for n in range(0, 8):
    if in_window[n]:
        windows[n].append((win_start[n], prev_t2))

for n in [7, 6, 5, 4, 3, 2, 1, 0]:
    total_h = sum(e-s for s,e in windows[n]) / 3600
    if total_h > 0.01:
        print('  {} cranes: {} windows, total {:.1f}h ({:.1f}% of time)'.format(
            n, len(windows[n]), total_h, total_h/(total_dur/3600)*100))
        if n >= 6 and len(windows[n]) > 0:
            # Show longest 3
            windows[n].sort(key=lambda x: x[1]-x[0], reverse=True)
            for i, (s, e) in enumerate(windows[n][:3]):
                dur = e - s
                if dur > 10:
                    print('    #{0}: {1} -> {2} ({3})'.format(
                        i+1, fmt(s), fmt(e), fmt(dur)))

# Busiest moment detail
print('\n' + '='*60)
print('Peak concurrency snapshot')
print('='*60)

max_conc = max(windows.keys())
if windows[max_conc]:
    best_s, best_e = max(windows[max_conc], key=lambda x: x[1]-x[0])
    mid_t = (best_s + best_e) / 2
    print('Peak: {} cranes at t={} ({} -> {})'.format(max_conc, fmt(mid_t), fmt(best_s), fmt(best_e)))
    for cid in range(1, 8):
        ct = tl[(tl[c_crane]==cid) & (tl[c_start]<=mid_t) & (tl[c_end]>=mid_t)]
        if len(ct) > 0:
            r = ct.iloc[0]
            print('  Crane{}: {} {} shelf{} z{} x{} [{}->{}]'.format(
                cid, r[c_type], r[c_order], r[c_shelf], r[c_z], r[c_x],
                fmt(r[c_start]), fmt(r[c_end])))
        else:
            print('  Crane{}: IDLE'.format(cid))

# Idle gap analysis
print('\n' + '='*60)
print('Idle gap analysis')
print('='*60)
for cid in range(1, 8):
    ct = tl[tl[c_crane]==cid].sort_values(c_start)
    long_idle = 0
    long_samples = []
    for i in range(1, len(ct)):
        gap = ct.iloc[i][c_start] - ct.iloc[i-1][c_end]
        if gap > 3600:
            long_idle += 1
            if len(long_samples) < 3:
                long_samples.append((ct.iloc[i-1][c_end], gap))
    samples_str = ''
    if long_samples:
        samples_str = ' e.g. ' + ', '.join(['t={} idle{}'.format(fmt(s), fmt(g)) for s,g in long_samples])
    print('  Crane{}: {} gaps >1h{}'.format(cid, long_idle, samples_str))

# Task distribution
print('\n' + '='*60)
print('Task distribution by aisle')
print('='*60)
for cid in range(1, 8):
    ct = tl[tl[c_crane]==cid]
    ob = (ct[c_type]=='outbound').sum()
    ib = (ct[c_type]=='inbound').sum()
    print('  Aisle{}(shelves{}-{}): out{} + in{} = {}'.format(
        cid, 2*cid-1, 2*cid, ob, ib, ob+ib))
