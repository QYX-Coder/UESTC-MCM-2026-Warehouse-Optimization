"""
问题三 可视化：并发图 + 甘特图 + 出入库分布 + 三维位置变化图
"""
import pandas as pd, numpy as np, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from collections import defaultdict

# ── Font ──
for fn in ['Microsoft YaHei', 'SimSun', 'SimHei']:
    try:
        fm.findfont(fn, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [fn, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception: continue

plt.rcParams.update({'font.size':10, 'axes.titlesize':13, 'axes.labelsize':11,
                     'axes.titleweight':'bold', 'axes.labelweight':'bold',
                     'figure.titlesize':15, 'figure.titleweight':'bold'})

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
OUT_DIR = os.path.join(ROOT, '输出结果', '问题三')
CSV_DIR = os.path.join(OUT_DIR, '调度结果CSV')
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════
tl = pd.read_csv(os.path.join(CSV_DIR, '各堆垛机作业时间表.csv'))
in_df = pd.read_csv(os.path.join(CSV_DIR, '入库货位分配方案.csv'))
out_df = pd.read_csv(os.path.join(CSV_DIR, '出库作业序列.csv'))
alloc_init = pd.read_csv(os.path.join(ROOT, '输出结果', '问题一', '分配结果CSV', '完整分配明细.csv'))

# Column names (handle encoding variations)
cols_tl = tl.columns.tolist()
cols_in = in_df.columns.tolist()
cols_out = out_df.columns.tolist()
cols_alloc = alloc_init.columns.tolist()

# Map columns: tl
c_crane = cols_tl[0]      # 堆垛机编号
c_lx = cols_tl[1]          # 巷道号
c_type = cols_tl[2]        # 任务类型
c_order = cols_tl[3]       # 订单号
c_mat = cols_tl[4]         # 原材料编号
c_box = cols_tl[5]         # 箱子类型
c_arrive = cols_tl[6]      # 到达时刻(s)
c_start = cols_tl[7]       # 开始时刻(s)
c_end = cols_tl[8]         # 完成时刻(s)
c_op = cols_tl[9]          # 操作时间(s)
c_wait = cols_tl[10]       # 等待时间(s)
c_flow = cols_tl[11]       # 流动时间(s)
c_shelf = cols_tl[12]      # 货架号
c_z = cols_tl[13]          # 层(z)
c_x = cols_tl[14]          # 列(x)
c_slot = cols_tl[15]       # 深浅位
c_compound = cols_tl[16]   # 是否复合

# Map columns: inbound
c_in_batch = cols_in[0]
c_in_mat = cols_in[1]
c_in_box = cols_in[2]
c_in_crane = cols_in[3]
c_in_shelf = cols_in[4]
c_in_z = cols_in[5]
c_in_x = cols_in[6]
c_in_slot = cols_in[7]
c_in_arrive = cols_in[8]
c_in_start = cols_in[9]
c_in_end = cols_in[10]
c_in_flow = cols_in[11]
c_in_compound = cols_in[12]

# Map columns: outbound
c_out_order = cols_out[0]
c_out_mat = cols_out[1]
c_out_box = cols_out[2]
c_out_crane = cols_out[3]
c_out_shelf = cols_out[4]
c_out_z = cols_out[5]
c_out_x = cols_out[6]
c_out_slot = cols_out[7]
c_out_arrive = cols_out[8]
c_out_start = cols_out[9]
c_out_end = cols_out[10]
c_out_op = cols_out[11]
c_out_wait = cols_out[12]
c_out_flow = cols_out[13]
c_out_compound = cols_out[14]

# Map columns: allocation — col order: shelf, 层(y), 层(z), 列(x), slot, mat, type...
c_a_shelf = cols_alloc[0]
c_a_z = cols_alloc[2]    # 层(z)
c_a_x = cols_alloc[3]    # 列(x)
c_a_slot_alloc = cols_alloc[4]  # 深浅位
c_a_mat = cols_alloc[5]
c_a_box = cols_alloc[6]

C_MAP = {'E1': '#2196F3', 'E3': '#FF9800', 'E4': '#4CAF50'}
TYPE_COLORS = {'outbound': '#E53935', 'inbound': '#43A047', 'compound': '#1E88E5'}
STATUS_COLORS = {'unchanged': '#BDBDBD', 'outbound': '#E53935',
                  'inbound': '#43A047', 'mixed': '#FF9800', 'empty': '#E8E8E8',
                  'outbound_only': '#E53935', 'inbound_only': '#43A047'}

def fmt_hms(s):
    h = int(s//3600); m = int((s%3600)//60)
    return f"{h}h{m:02d}m"

# ═══════════════════════════════════════════════════════════
# FIGURE 1: Concurrency analysis
# ═══════════════════════════════════════════════════════════
print("Fig1: Concurrency analysis...")
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Compute concurrency data first (shared)
events = []
for _, row in tl.iterrows():
    events.append((row[c_start], +1))
    events.append((row[c_end], -1))
events.sort()

active = 0; conc_total = {}; prev_t = events[0][0]
for t, delta in events:
    dur = t - prev_t
    if dur > 0:
        conc_total[active] = conc_total.get(active, 0) + dur
    active += delta
    prev_t = t

total_dur = sum(conc_total.values())

# 1a: Utilization bar chart
ax = axes[0]
util_data = []
for cid in range(1, 8):
    ct = tl[tl[c_crane]==cid]
    busy = ct[c_op].sum()
    span = ct[c_end].max() - ct[c_start].min()
    util = busy / span * 100
    n = len(ct)
    util_data.append((cid, n, busy/3600, span/3600, util))

ids = [d[0] for d in util_data]
utils = [d[4] for d in util_data]
tasks = [d[1] for d in util_data]
bars = ax.bar(range(7), utils, color=plt.cm.YlOrRd([u/50 for u in utils]), edgecolor='#333', linewidth=0.8)
for i, (cid, n, busy_h, span_h, u) in enumerate(util_data):
    ax.text(i, u+0.8, f'{u:.1f}%\n{n}任务', ha='center', fontsize=12, fontweight='bold')
ax.set_xticks(range(7))
ax.set_xticklabels([f'堆垛机{i+1}\n(巷道{i+1})' for i in range(7)], fontsize=11, fontweight='bold')
ax.set_ylabel('利用率 (%)', fontsize=14, fontweight='bold')
ax.set_title('各堆垛机利用率', fontsize=16, fontweight='bold')
ax.set_ylim(0, 58)
ax.tick_params(axis='y', labelsize=11)
ax.grid(axis='y', alpha=0.3)

# 1b: Concurrency time-weighted distribution (merged 3+)
ax = axes[1]
# Merge 3-7 into "≥3台"
merged_data = {0: conc_total.get(0, 0), 1: conc_total.get(1, 0),
               2: conc_total.get(2, 0),
               3: sum(conc_total.get(k, 0) for k in range(3, 8))}
merged_labels = ['0台', '1台', '2台', '≥3台\n(多台并发)']
merged_colors = ['#E8E8E8', '#BBDEFB', '#90CAF9', '#42A5F5']
merged_hours = [merged_data[k]/3600 for k in [0, 1, 2, 3]]

x_pos = np.arange(4)
bars = ax.bar(x_pos, merged_hours, color=merged_colors, edgecolor='#333', linewidth=1.0)
for i, (k, h) in enumerate(zip([0, 1, 2, 3], merged_hours)):
    pct = merged_data[k]/total_dur*100
    # Show label for all bars including 7th (now merged into 3+)
    ax.text(i, h+0.8, f'{h:.1f}h\n({pct:.1f}%)', ha='center', fontsize=13, fontweight='bold')

# Annotate 5+ concurrency within the merged bar
high_conc = sum(conc_total.get(k, 0) for k in [5, 6, 7])
high_h = high_conc / 3600
high_pct = high_conc / total_dur * 100
ax.annotate(f'其中≥5台并发:\n{high_h:.1f}h ({high_pct:.1f}%)',
            xy=(3, merged_hours[3]), xytext=(3.5, merged_hours[3]*0.85),
            fontsize=14, fontweight='bold', color='#1565C0',
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2.5),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', edgecolor='#1565C0', lw=2))

ax.set_xticks(x_pos)
ax.set_xticklabels(merged_labels, fontsize=12, fontweight='bold')
ax.set_xlabel('同时工作堆垛机数', fontsize=14, fontweight='bold')
ax.set_ylabel('时长 (h)', fontsize=14, fontweight='bold')
ax.set_title('并发度时间加权分布', fontsize=16, fontweight='bold')
ax.tick_params(axis='y', labelsize=11)
ax.set_ylim(0, max(merged_hours)*1.2)
ax.grid(axis='y', alpha=0.3)

# 1c: Concurrency summary table
ax = axes[2]
ax.axis('off')

# Detailed breakdown + merged row
table_data = []
for n in [7, 6, 5, 4, 3, 2, 1, 0]:
    h = conc_total.get(n, 0)/3600
    pct = conc_total.get(n, 0)/total_dur*100
    table_data.append([f'{n}台并发', f'{h:.1f}h', f'{pct:.1f}%'])

# Add summary rows
h_35 = sum(conc_total.get(k, 0) for k in [3, 4, 5, 6, 7])/3600
pct_35 = sum(conc_total.get(k, 0) for k in [3, 4, 5, 6, 7])/total_dur*100
h_57 = sum(conc_total.get(k, 0) for k in [5, 6, 7])/3600
pct_57 = sum(conc_total.get(k, 0) for k in [5, 6, 7])/total_dur*100
h_67 = sum(conc_total.get(k, 0) for k in [6, 7])/3600
pct_67 = sum(conc_total.get(k, 0) for k in [6, 7])/total_dur*100

table_data.append(['───', '───', '───'])
table_data.append(['≥3台合计', f'{h_35:.1f}h', f'{pct_35:.1f}%'])
table_data.append(['≥5台合计', f'{h_57:.1f}h', f'{pct_57:.1f}%'])
table_data.append(['≥6台合计', f'{h_67:.1f}h', f'{pct_67:.1f}%'])

table = ax.table(cellText=table_data,
                 colLabels=['并发数', '累计时长', '时间占比'],
                 cellLoc='center', loc='center',
                 colWidths=[0.28, 0.28, 0.28])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.4, 1.7)
for key, cell in table.get_celld().items():
    cell.set_linewidth(1.0)
    if key[0] == 0:
        # Header row
        cell.set_facecolor('#455A64')
        cell.set_text_props(color='white', fontweight='bold', fontsize=13)
    elif key[0] in [9, 10, 11, 12]:
        # Summary rows
        cell.set_facecolor('#E3F2FD')
        cell.set_text_props(fontweight='bold', fontsize=12, color='#1565C0')
    elif key[0] in [1, 2, 3]:
        # 7, 6, 5 cranes
        cell.set_facecolor('#BBDEFB')
    else:
        cell.set_facecolor('#FAFAFA')

ax.set_title('七机并发统计汇总', fontsize=16, fontweight='bold', y=0.98)

fig.suptitle('问题三：七台堆垛机并发运作分析', fontsize=19, fontweight='bold', y=1.01)
plt.tight_layout(pad=2.5, w_pad=3)
fig.savefig(os.path.join(OUT_DIR, '并发分析图.png'), dpi=150, bbox_inches='tight', pad_inches=0.3)
plt.close()
print("  -> 并发分析图.png")

# ═══════════════════════════════════════════════════════════
# FIGURE 2: Crane operation Gantt chart
# ═══════════════════════════════════════════════════════════
print("Fig2: Crane Gantt chart...")

fig, axes = plt.subplots(2, 1, figsize=(22, 14), gridspec_kw={'height_ratios': [1, 2]})

# 2a: Full timeline overview (crane utilization by time blocks)
ax = axes[0]
block_h = 2  # 2-hour blocks
max_t = tl[c_end].max()
n_blocks = int(np.ceil(max_t / 3600 / block_h))

# Count active tasks per block per crane
heatmap_data = np.zeros((7, n_blocks))
for cid in range(1, 8):
    ct = tl[tl[c_crane]==cid]
    for _, row in ct.iterrows():
        b0 = int(row[c_start] // (3600*block_h))
        b1 = int(row[c_end] // (3600*block_h))
        for b in range(b0, min(b1+1, n_blocks)):
            t0_block = b * 3600 * block_h
            t1_block = (b+1) * 3600 * block_h
            overlap = min(row[c_end], t1_block) - max(row[c_start], t0_block)
            if overlap > 0:
                heatmap_data[cid-1, b] += overlap / (3600*block_h)

im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1,
               extent=[0, n_blocks*block_h, 7.5, 0.5])
ax.set_yticks(range(1, 8))
ax.set_yticklabels([f'堆垛机{i}' for i in range(1, 8)], fontsize=9)
ax.set_xlabel('仿真时间 (h)', fontsize=11, fontweight='bold')
ax.set_title(f'堆垛机作业热力图 ({block_h}h粒度, 颜色=该时段忙碌占比)', fontsize=13, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('忙碌占比', fontsize=10)

# Mark order arrival density
arrival_times = sorted(list(tl[c_arrive].unique()))
arrival_hist, bin_edges = np.histogram(arrival_times, bins=n_blocks, range=(0, n_blocks*3600*block_h))
ax2 = ax.twiny()
ax2.fill_between(np.arange(n_blocks)*block_h + block_h/2, arrival_hist, alpha=0.15, color='blue')
ax2.set_xlim(ax.get_xlim())
ax2.set_xlabel('', fontsize=0)

# 2b: Zoomed Gantt (busiest 48-hour window)
ax = axes[1]
# Find the busiest 48h window
window_h = 48
best_start = 0; best_busy = 0
for t0 in np.arange(0, max_t - window_h*3600, 3600):
    busy_sum = 0
    for cid in range(1, 8):
        ct = tl[tl[c_crane]==cid]
        mask = (ct[c_end] > t0) & (ct[c_start] < t0 + window_h*3600)
        busy_sum += ct[mask][c_op].sum()
    if busy_sum > best_busy:
        best_busy = busy_sum
        best_start = t0

t0_window = best_start
t1_window = t0_window + window_h * 3600

color_op = {'outbound': '#E53935', 'inbound': '#43A047'}
for cid in range(1, 8):
    ct = tl[(tl[c_crane]==cid) & (tl[c_end] > t0_window) & (tl[c_start] < t1_window)]
    for _, row in ct.iterrows():
        t_s = max(row[c_start], t0_window) / 3600
        t_e = min(row[c_end], t1_window) / 3600
        op_type = 'outbound' if row[c_type] == 'outbound' else 'inbound'
        is_compound = str(row[c_compound]).strip() == '是'
        edge_color = '#FFD600' if is_compound else 'none'
        edge_w = 0.5 if is_compound else 0
        ax.barh(cid, (t_e - t_s), left=t_s, height=0.7,
                color=color_op.get(op_type, '#999'), edgecolor=edge_color,
                linewidth=edge_w, alpha=0.85)

ax.set_yticks(range(1, 8))
ax.set_yticklabels([f'堆垛机{i}' for i in range(1, 8)], fontsize=9)
ax.set_xlabel(f'仿真时间 (h, 从t={fmt_hms(t0_window)}起)', fontsize=11, fontweight='bold')
ax.set_title(f'堆垛机作业甘特图 (最繁忙{window_h}h窗口: {fmt_hms(t0_window)} → {fmt_hms(t1_window)})',
             fontsize=13, fontweight='bold')
ax.set_xlim(t0_window/3600, t1_window/3600)
ax.grid(axis='x', alpha=0.3)

legend_elements = [
    Patch(facecolor='#E53935', label='纯出库'),
    Patch(facecolor='#43A047', label='纯入库'),
    Patch(facecolor='#E53935', edgecolor='#FFD600', linewidth=1.5, label='复合(出库部分)'),
    Patch(facecolor='#43A047', edgecolor='#FFD600', linewidth=1.5, label='复合(入库部分)'),
]
ax.legend(handles=legend_elements, fontsize=8, ncol=4, loc='upper right')

fig.suptitle('问题三：堆垛机作业时间表', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(pad=3)
fig.savefig(os.path.join(OUT_DIR, '堆垛机作业甘特图.png'), dpi=150, bbox_inches='tight', pad_inches=0.3)
plt.close()
print("  -> 堆垛机作业甘特图.png")

# ═══════════════════════════════════════════════════════════
# FIGURE 3: Inbound + Outbound distribution + task mix
# ═══════════════════════════════════════════════════════════
print("Fig3: Inbound/outbound distribution...")

fig, axes = plt.subplots(2, 3, figsize=(22, 14))

# 3a: Inbound positions heatmap (z vs x, all shelves aggregated)
ax = axes[0, 0]
in_grid = np.zeros((50, 68))
for _, row in in_df.iterrows():
    in_grid[int(row[c_in_z])-1, int(row[c_in_x])-1] += 1
im = ax.imshow(in_grid, aspect='auto', cmap='Greens', origin='lower',
               extent=[0.5, 68.5, 0.5, 50.5])
ax.axhline(8.5, color='#0077BB', lw=1.5, ls='--')
ax.axhline(42.5, color='#CC3311', lw=1.5, ls='--')
ax.set_xlabel('列 x', fontsize=11, fontweight='bold')
ax.set_ylabel('层 z', fontsize=11, fontweight='bold')
ax.set_title(f'入库货位分配分布 ({len(in_df)}箱)', fontsize=12, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('入库箱数', fontsize=9)

# 3b: Outbound positions heatmap
ax = axes[0, 1]
out_grid = np.zeros((50, 68))
for _, row in out_df.iterrows():
    out_grid[int(row[c_out_z])-1, int(row[c_out_x])-1] += 1
im = ax.imshow(out_grid, aspect='auto', cmap='Reds', origin='lower',
               extent=[0.5, 68.5, 0.5, 50.5])
ax.axhline(8.5, color='#0077BB', lw=1.5, ls='--')
ax.axhline(42.5, color='#CC3311', lw=1.5, ls='--')
ax.set_xlabel('列 x', fontsize=11, fontweight='bold')
ax.set_ylabel('层 z', fontsize=11, fontweight='bold')
ax.set_title(f'出库取货位置分布 ({len(out_df)}单)', fontsize=12, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('出库单数', fontsize=9)

# 3c: Operation type by aisle
ax = axes[0, 2]
aisle_stats = {}
for cid in range(1, 8):
    ct = tl[tl[c_crane]==cid]
    ob = (ct[c_type]=='outbound').sum()
    ib = (ct[c_type]=='inbound').sum()
    aisle_stats[cid] = {'outbound': ob, 'inbound': ib}

x_pos = np.arange(7)
w = 0.35
ob_vals = [aisle_stats[i+1]['outbound'] for i in range(7)]
ib_vals = [aisle_stats[i+1]['inbound'] for i in range(7)]
ax.bar(x_pos - w/2, ob_vals, w,
       color='#E53935', edgecolor='#333', linewidth=0.6, label='出库')
ax.bar(x_pos + w/2, ib_vals, w,
       color='#43A047', edgecolor='#333', linewidth=0.6, label='入库')
for i in range(7):
    total = ob_vals[i] + ib_vals[i]
    ax.text(i, total+35, str(total), ha='center', va='bottom', fontsize=11, fontweight='bold')
max_total = max(ob_vals[i] + ib_vals[i] for i in range(7))
ax.set_ylim(0, max_total * 1.18)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'巷道{i+1}' for i in range(7)], fontsize=10)
ax.set_ylabel('操作次数', fontsize=12, fontweight='bold')
ax.set_title('各巷道出入库操作分布', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# 3d: Box type distribution in inbound
ax = axes[1, 0]
in_box_counts = in_df[c_in_box].value_counts()
box_order = ['E1', 'E3', 'E4']
colors_box = [C_MAP[b] for b in box_order]
counts_box = [in_box_counts.get(b, 0) for b in box_order]
wedges, texts, autotexts = ax.pie(counts_box, labels=box_order, autopct='%1.1f%%',
                                    colors=colors_box, startangle=90,
                                    textprops={'fontsize': 11, 'fontweight': 'bold'})
for at in autotexts:
    at.set_fontsize(10)
ax.set_title(f'入库箱型分布 ({len(in_df)}箱)', fontsize=12, fontweight='bold')

# 3e: Operation type distribution (pure vs compound)
ax = axes[1, 1]
pure_out = ((tl[c_type]=='outbound') & (tl[c_compound].astype(str).str.strip()=='否')).sum()
pure_in = ((tl[c_type]=='inbound') & (tl[c_compound].astype(str).str.strip()=='否')).sum()
compound = (tl[c_compound].astype(str).str.strip()=='是').sum()

op_labels = ['纯出库', '纯入库', '复合作业']
op_counts = [pure_out, pure_in, compound]
op_colors = ['#E53935', '#43A047', '#1E88E5']
wedges, texts, autotexts = ax.pie(op_counts, labels=op_labels, autopct='%1.1f%%',
                                    colors=op_colors, startangle=90,
                                    textprops={'fontsize': 11, 'fontweight': 'bold'})
for at in autotexts:
    at.set_fontsize(10)
ax.set_title(f'操作类型分布 ({len(tl)}次)', fontsize=12, fontweight='bold')

# 3f: Flow time distribution by operation type
ax = axes[1, 2]
flow_data = []
labels_flow = []
for op_type, color in [('纯出库', '#E53935'), ('纯入库', '#43A047'), ('复合操作', '#1E88E5')]:
    if '纯出库' in op_type:
        mask = (tl[c_type]=='outbound') & (tl[c_compound].astype(str).str.strip()=='否')
    elif '纯入库' in op_type:
        mask = (tl[c_type]=='inbound') & (tl[c_compound].astype(str).str.strip()=='否')
    else:
        mask = tl[c_compound].astype(str).str.strip()=='是'
    flows = tl.loc[mask, c_flow].values / 3600
    if len(flows) > 0:
        parts = ax.violinplot(flows, positions=[len(flow_data)], showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        flow_data.append(flows)
        labels_flow.append(f'{op_type}\n(n={len(flows)}, med={np.median(flows):.1f}h)')

ax.set_xticks(range(len(labels_flow)))
ax.set_xticklabels(labels_flow, fontsize=9)
ax.set_ylabel('流动时间 (h)', fontsize=11, fontweight='bold')
ax.set_title('各操作类型流动时间分布', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

fig.suptitle('问题三：出入库作业序列与分布', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout(pad=3)
fig.savefig(os.path.join(OUT_DIR, '出入库分布图.png'), dpi=150, bbox_inches='tight', pad_inches=0.3)
plt.close()
print("  -> 出入库分布图.png")

# ═══════════════════════════════════════════════════════════
# FIGURE 4: 3D cube - position change status
# ═══════════════════════════════════════════════════════════
print("Fig4: 3D position change cube...")

# Build position status map
# Initial state: all 65000 occupied positions
N_S, N_L, N_C = 14, 50, 68
initial_occupied = {}  # (shelf, z, x, slot) -> material_id
for _, row in alloc_init.iterrows():
    key = (int(row[c_a_shelf]), int(row[c_a_z]), int(row[c_a_x]), str(row[c_a_slot_alloc]).strip())
    initial_occupied[key] = row[c_a_mat]

# Outbound: positions that were freed
outbound_positions = set()
for _, row in out_df.iterrows():
    key = (int(row[c_out_shelf]), int(row[c_out_z]), int(row[c_out_x]), str(row[c_out_slot]).strip())
    outbound_positions.add(key)

# Inbound: positions that were filled
inbound_positions = set()
for _, row in in_df.iterrows():
    key = (int(row[c_in_shelf]), int(row[c_in_z]), int(row[c_in_x]), str(row[c_in_slot]).strip())
    inbound_positions.add(key)

# Classify every position
unchanged_set = set()
outbound_only_set = set()
inbound_only_set = set()
mixed_set = set()
empty_set = set()

all_positions = set()
for shelf in range(1, N_S+1):
    for z in range(1, N_L+1):
        for x in range(1, N_C+1):
            for slot in ['浅位', '深位']:
                all_positions.add((shelf, z, x, slot))

for pos in all_positions:
    was_occupied = pos in initial_occupied
    was_outbound = pos in outbound_positions
    was_inbound = pos in inbound_positions
    if was_outbound and was_inbound:
        mixed_set.add(pos)
    elif was_outbound:
        outbound_only_set.add(pos)
    elif was_inbound:
        inbound_only_set.add(pos)
    elif was_occupied:
        unchanged_set.add(pos)
    else:
        empty_set.add(pos)

print(f"  未调整: {len(unchanged_set)}, 纯出库: {len(outbound_only_set)}, "
      f"纯入库: {len(inbound_only_set)}, 混合: {len(mixed_set)}, 始终空: {len(empty_set)}")

# ── 4a: 3D scatter (aggregated by x,z across all shelves) ──
fig = plt.figure(figsize=(28, 12))

ax = fig.add_subplot(1, 2, 1, projection='3d')
# Downsample for 3D: show only positions with changes
all_changed = mixed_set | outbound_only_set | inbound_only_set
# Aggregate to (x,z) grid counting by shelf
grid_status = {}  # (x,z) -> {'unchanged':N, 'outbound':N, 'inbound':N, 'mixed':N, 'empty':N}
for pos in all_positions:
    shelf, z, x, slot = pos
    gkey = (x, z)
    if gkey not in grid_status:
        grid_status[gkey] = {'unchanged': 0, 'outbound': 0, 'inbound': 0, 'mixed': 0, 'empty': 0}
    if pos in mixed_set:
        grid_status[gkey]['mixed'] += 1
    elif pos in outbound_only_set:
        grid_status[gkey]['outbound'] += 1
    elif pos in inbound_only_set:
        grid_status[gkey]['inbound'] += 1
    elif pos in unchanged_set:
        grid_status[gkey]['unchanged'] += 1
    else:
        grid_status[gkey]['empty'] += 1

# Scatter: each point is an (x,z) position, height = count, color = dominant status
xs, zs, heights = [], [], []
colors_3d = []
max_h = 0
for (x, z), counts in grid_status.items():
    total = sum(counts.values())
    if total > 0:
        xs.append(x)
        zs.append(z)
        heights.append(total/2)  # scale down for visualization
        max_h = max(max_h, total)
        # Determine dominant status
        dom = max(counts, key=counts.get)
        colors_3d.append(STATUS_COLORS[dom])

sc = ax.bar3d(np.array(xs)-0.5, np.array(zs)-0.5, np.zeros(len(xs)),
              0.9, 0.9, heights, color=colors_3d, alpha=0.7, edgecolor='none', shade=True)
ax.set_xlabel('列 x', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel('层 z', fontsize=16, fontweight='bold', labelpad=10)
ax.set_zlabel('位置数', fontsize=16, fontweight='bold', labelpad=10)
ax.set_title(f'(x,z)网格位置变化统计\n(每格最多28个slot=14货架x2深浅位)', fontsize=18, fontweight='bold', pad=20)
ax.view_init(elev=25, azim=-45)
ax.tick_params(axis='both', labelsize=13)
# Suppress z-axis ticks to avoid floating number clutter
ax.zaxis.set_tick_params(labelsize=12)
ax.set_zticks([0, 5, 10, 15])
ax.set_zticklabels(['0', '5', '10', '15'], fontsize=12)

legend_elements_3d = [
    Patch(facecolor=STATUS_COLORS['unchanged'], label=f'未调整 ({len(unchanged_set)})'),
    Patch(facecolor=STATUS_COLORS['outbound_only'], label=f'纯出库 ({len(outbound_only_set)})'),
    Patch(facecolor=STATUS_COLORS['inbound_only'], label=f'纯入库 ({len(inbound_only_set)})'),
    Patch(facecolor=STATUS_COLORS['mixed'], label=f'出入库混合 ({len(mixed_set)})'),
]
ax.legend(handles=legend_elements_3d, fontsize=13, loc='upper left', markerscale=1.5)

# ── 4b: 2D heatmap of change type by (x,z) ──
ax2 = fig.add_subplot(1, 2, 2)

# Build a grid showing dominant status per (x,z)
change_grid = np.full((N_L, N_C), -1)  # -1=empty
status_to_code = {'unchanged': 0, 'outbound': 1, 'inbound': 2, 'mixed': 3, 'empty': -1}
for (x, z), counts in grid_status.items():
    total = sum(counts.values())
    if total > 0:
        dom = max(counts, key=counts.get)
        change_grid[z-1, x-1] = status_to_code[dom]

cmap_colors = ['#E8E8E8', '#BDBDBD', '#E53935', '#43A047', '#FF9800']
cmap = ListedColormap(cmap_colors)
im = ax2.imshow(change_grid, aspect='auto', cmap=cmap, origin='lower',
               extent=[0.5, N_C+0.5, 0.5, N_L+0.5], vmin=-1.5, vmax=3.5)
ax2.axhline(8.5, color='#0077BB', lw=2, ls='--', alpha=0.8)
ax2.axhline(42.5, color='#CC3311', lw=2, ls='--', alpha=0.8)
ax2.set_xlabel('列 x', fontsize=16, fontweight='bold', labelpad=8)
ax2.set_ylabel('层 z', fontsize=16, fontweight='bold', labelpad=8)
ax2.set_title('(x,z)位置主导变化类型\n(按14货架汇总)', fontsize=18, fontweight='bold', pad=12)
ax2.tick_params(axis='both', labelsize=13)

legend_elements_2d = [
    Patch(facecolor='#E8E8E8', label='始终空置'),
    Patch(facecolor='#BDBDBD', label='未调整'),
    Patch(facecolor='#E53935', label='纯出库'),
    Patch(facecolor='#43A047', label='纯入库'),
    Patch(facecolor='#FF9800', label='出入库混合'),
]
ax2.legend(handles=legend_elements_2d, fontsize=13, loc='lower right',
          markerscale=1.5)

fig.suptitle('问题三：仓库位置变化三维统计图', fontsize=22, fontweight='bold', y=1.01)
plt.subplots_adjust(left=0.05, right=0.92, top=0.92, bottom=0.08, wspace=0.25)
fig.savefig(os.path.join(OUT_DIR, '位置变化三维图.png'), dpi=150, bbox_inches='tight', pad_inches=0.3)
plt.close()
print("  -> 位置变化三维图.png")

# ═══════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Position change summary:")
print(f"  未调整 (unchanged):     {len(unchanged_set):>6}")
print(f"  纯出库 (outbound only):  {len(outbound_only_set):>6}")
print(f"  纯入库 (inbound only):   {len(inbound_only_set):>6}")
print(f"  出入库混合 (mixed):      {len(mixed_set):>6}")
print(f"  始终空置 (always empty): {len(empty_set):>6}")
print(f"  总计:                    {len(all_positions):>6}")
print(f"  验证: {len(unchanged_set)}+{len(outbound_only_set)}+{len(inbound_only_set)}+{len(mixed_set)}+{len(empty_set)} = {len(unchanged_set)+len(outbound_only_set)+len(inbound_only_set)+len(mixed_set)+len(empty_set)}")
print(f"  初始65000 + 入库5069 - 出库3738 = {65000+5069-3738}")
print(f"  实际最终占用 = {len(unchanged_set)+len(inbound_only_set)+len(mixed_set)}")

print("\nAll figures saved to:", OUT_DIR)
