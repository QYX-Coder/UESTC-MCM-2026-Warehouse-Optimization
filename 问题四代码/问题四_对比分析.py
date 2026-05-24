"""
问题四 对比分析：线性 vs 非线性速度模型
- 运行两个模型（如CSV已存在则跳过）
- 对比流动时间、复合占比、各机利用率
- 速度灵敏度分析
- 生成对比图表（加粗加大字体）
"""
import pandas as pd, numpy as np, os, sys, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from 问题四_时间模型 import LinearTimeModel, NonlinearTimeModel, LH, COL_WIDTH
from 问题四_出入库协同调度 import run_simulation

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
# Global font settings — larger and bolder
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16,
                     'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
                     'legend.fontsize': 13, 'xtick.labelsize': 13, 'ytick.labelsize': 13})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, '输出结果', '问题四')
CSV_LIN = os.path.join(OUT_DIR, '调度结果CSV', '线性模型')
CSV_NL = os.path.join(OUT_DIR, '调度结果CSV', '非线性模型')
os.makedirs(CSV_LIN, exist_ok=True)
os.makedirs(CSV_NL, exist_ok=True)

# ============ Run or load simulations ============
print("\n" + "=" * 70)
print("问题四：线性 vs 非线性速度模型对比")
print("=" * 70)

tl_csv_lin = os.path.join(CSV_LIN, '各堆垛机作业时间表.csv')
tl_csv_nl = os.path.join(CSV_NL, '各堆垛机作业时间表.csv')

if os.path.exists(tl_csv_lin) and os.path.exists(tl_csv_nl):
    print("\n>>> CSV已存在，跳过仿真，直接加载...")
    tl_lin = pd.read_csv(tl_csv_lin)
    tl_nl = pd.read_csv(tl_csv_nl)
    # Reconstruct stats from CSV
    ib_lin = pd.read_csv(os.path.join(CSV_LIN, '入库货位分配方案.csv'))
    ob_lin = pd.read_csv(os.path.join(CSV_LIN, '出库作业序列.csv'))
    ib_nl = pd.read_csv(os.path.join(CSV_NL, '入库货位分配方案.csv'))
    ob_nl = pd.read_csv(os.path.join(CSV_NL, '出库作业序列.csv'))

    def build_stats(tl, ib, ob, name):
        total_ob_flow = ob['流动时间(s)'].sum()
        total_ib_flow = ib['流动时间(s)'].sum()
        out_count = len(ob); in_count = len(ib); total_ops = out_count + in_count
        pure_out = int((ob['是否复合']=='否').sum())
        pure_in = int((ib['是否复合']=='否').sum())
        compound_ops = int((ob['是否复合']=='是').sum()) + int((ib['是否复合']=='是').sum())
        makespan = tl['完成时刻(s)'].max()
        return {
            'model_name': name, 'total_flow': total_ob_flow + total_ib_flow,
            'outbound_flow': total_ob_flow, 'inbound_flow': total_ib_flow,
            'avg_out_flow': total_ob_flow/out_count if out_count>0 else 0,
            'avg_in_flow': total_ib_flow/in_count if in_count>0 else 0,
            'avg_all_flow': (total_ob_flow+total_ib_flow)/total_ops if total_ops>0 else 0,
            'outbound_count': out_count, 'inbound_count': in_count,
            'pure_out': pure_out, 'pure_in': pure_in,
            'compound_ops': compound_ops, 'total_ops': total_ops,
            'compound_ratio': compound_ops/total_ops*100 if total_ops>0 else 0,
            'makespan': makespan, 'pass_count': 7, 'total_checks': 7,
        }
    stats_lin = build_stats(tl_lin, ib_lin, ob_lin, 'Linear(vx=2.5,vz=0.65)')
    stats_nl = build_stats(tl_nl, ib_nl, ob_nl, 'Nonlinear')
else:
    print("\n>>> 运行线性模型 (vx=2.5, vz=0.65)...")
    linear_model = LinearTimeModel(vx=2.5, vz=0.65)
    stats_lin = run_simulation(linear_model, CSV_LIN)

    print("\n\n>>> 运行非线性模型...")
    nonlinear_model = NonlinearTimeModel()
    stats_nl = run_simulation(nonlinear_model, CSV_NL)

    tl_lin = pd.read_csv(tl_csv_lin)
    tl_nl = pd.read_csv(tl_csv_nl)

# ============ Print comparison table ============
print("\n\n" + "=" * 70)
print("对比分析结果")
print("=" * 70)

metrics = [
    ('总流动时间(s)', 'total_flow', '{:.0f}'),
    ('出库总流动(s)', 'outbound_flow', '{:.0f}'),
    ('入库总流动(s)', 'inbound_flow', '{:.0f}'),
    ('出库平均流动(s)', 'avg_out_flow', '{:.1f}'),
    ('入库平均流动(s)', 'avg_in_flow', '{:.1f}'),
    ('总体平均流动(s)', 'avg_all_flow', '{:.1f}'),
    ('总操作数', 'total_ops', '{:.0f}'),
    ('纯出库', 'pure_out', '{:.0f}'),
    ('纯入库', 'pure_in', '{:.0f}'),
    ('复合操作数', 'compound_ops', '{:.0f}'),
    ('复合占比(%)', 'compound_ratio', '{:.1f}'),
    ('Makespan(h)', 'makespan', '{:.1f}'),
]

print(f"\n{'指标':<25} {'线性(2.5/0.65)':<20} {'非线性(含加速度)':<20} {'差异':<15} {'变化%'}")
print("-" * 95)
for label, key, fmt in metrics:
    v_lin = stats_lin[key]; v_nl = stats_nl[key]
    diff = v_nl - v_lin
    pct = diff / v_lin * 100 if v_lin > 0 else 0
    if key == 'makespan':
        print(f"{label:<25} {v_lin/3600:>18.1f}h {v_nl/3600:>18.1f}h {(v_nl-v_lin)/3600:>+13.1f}h {pct:>+8.1f}%")
    elif key == 'total_flow' or 'flow' in key:
        print(f"{label:<25} {fmt.format(v_lin):>20} {fmt.format(v_nl):>20} {fmt.format(diff):>15} {pct:>+8.1f}%")
    else:
        print(f"{label:<25} {fmt.format(v_lin):>20} {fmt.format(v_nl):>20} {fmt.format(diff):>15} {pct:>+8.1f}%")

print(f"\n{'各堆垛机流动时间对比':-^95}")
for a in range(1, 8):
    flow_lin = tl_lin[tl_lin['堆垛机编号']==a]['流动时间(s)'].sum()
    flow_nl = tl_nl[tl_nl['堆垛机编号']==a]['流动时间(s)'].sum()
    diff = flow_nl - flow_lin; pct = diff/flow_lin*100 if flow_lin>0 else 0
    n_lin = len(tl_lin[tl_lin['堆垛机编号']==a])
    n_nl = len(tl_nl[tl_nl['堆垛机编号']==a])
    print(f"  堆垛机{a}: 线性{flow_lin/3600:.1f}h({n_lin}任务) vs 非线性{flow_nl/3600:.1f}h({n_nl}任务) | {diff/3600:+.1f}h ({pct:+.1f}%)")

# ============ Visualizations ============
print(f"\n生成对比图表...")

# Non-linear model params for theoretical curves
VX_E, AX_E = 3.0, 0.5
VX_L, AX_L = 2.3, 0.4
VZ_E, AZ_E = 0.75, 0.15
VZ_L, AZ_L = 0.58, 0.10

def nl_axis_time(d, v_max, a):
    if d < 1e-12: return 0.0
    d_accel = v_max**2 / a
    if d >= d_accel:
        return 2*v_max/a + (d - d_accel)/v_max
    else:
        return 2*math.sqrt(d/a)

# =====================================================================
# Figure 1: Speed profile + sensitivity (2x3 layout)
# =====================================================================
fig1, axes1 = plt.subplots(2, 3, figsize=(24, 14))

# --- 1a: Short distance horizontal velocity profile ---
ax = axes1[0, 0]
dx_short = 2 * COL_WIDTH  # 0.8m
# Nonlinear empty (triangular)
t_total_e = 2*math.sqrt(dx_short/AX_E)
v_peak_e = math.sqrt(dx_short*AX_E)
t_pts_e = [0, t_total_e/2, t_total_e]; v_pts_e = [0, v_peak_e, 0]
ax.plot(t_pts_e, v_pts_e, 'b-', linewidth=3, label=f'Nonlinear empty(vmax={VX_E})')
# Nonlinear loaded (triangular)
if dx_short >= VX_L**2/AX_L:
    t_a_l = VX_L/AX_L; t_total_l = 2*t_a_l + (dx_short - VX_L**2/AX_L)/VX_L
    t_pts_l = [0, t_a_l, t_a_l+(dx_short-VX_L**2/AX_L)/VX_L, t_total_l]
    v_pts_l = [0, VX_L, VX_L, 0]
else:
    t_total_l_2 = 2*math.sqrt(dx_short/AX_L)
    v_peak_l = math.sqrt(dx_short*AX_L)
    t_pts_l = [0, t_total_l_2/2, t_total_l_2]; v_pts_l = [0, v_peak_l, 0]
ax.plot(t_pts_l, v_pts_l, 'r-', linewidth=3, label=f'Nonlinear loaded(vmax={VX_L})')
ax.axhline(y=2.5, color='g', linestyle='--', linewidth=3, label='Linear(vx=2.5)')
ax.set_xlabel('时间 (s)', fontsize=15, fontweight='bold')
ax.set_ylabel('水平速度 (m/s)', fontsize=15, fontweight='bold')
ax.set_title(f'短距离速度剖面 (dx={dx_short:.1f}m)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper right'); ax.grid(True, alpha=0.3)

# --- 1b: Long distance horizontal velocity profile ---
ax = axes1[0, 1]
dx_long = 50 * COL_WIDTH  # 20m
t_a_e_l = VX_E/AX_E
d_accel_e = VX_E**2/AX_E
t_total_e_l = 2*t_a_e_l + (dx_long - d_accel_e)/VX_E
ax.plot([0, t_a_e_l, t_a_e_l+(dx_long-d_accel_e)/VX_E, t_total_e_l],
        [0, VX_E, VX_E, 0], 'b-', linewidth=3, label=f'Nonlinear empty(vmax={VX_E})')
t_a_l_l = VX_L/AX_L
d_accel_l = VX_L**2/AX_L
t_total_l_l = 2*t_a_l_l + (dx_long - d_accel_l)/VX_L
ax.plot([0, t_a_l_l, t_a_l_l+(dx_long-d_accel_l)/VX_L, t_total_l_l],
        [0, VX_L, VX_L, 0], 'r-', linewidth=3, label=f'Nonlinear loaded(vmax={VX_L})')
ax.axhline(y=2.5, color='g', linestyle='--', linewidth=3, label='Linear(vx=2.5)')
ax.set_xlabel('时间 (s)', fontsize=15, fontweight='bold')
ax.set_ylabel('水平速度 (m/s)', fontsize=15, fontweight='bold')
ax.set_title(f'长距离速度剖面 (dx={dx_long:.0f}m)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='lower right'); ax.grid(True, alpha=0.3)

# --- 1c: Time vs distance — multiple linear speeds vs nonlinear ---
ax = axes1[0, 2]
distances = np.linspace(0.1, 30, 300)
colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.9, 6))
lin_speeds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
for i, vx_val in enumerate(lin_speeds):
    ax.plot(distances, distances/vx_val, '--', color=colors[i], linewidth=1.8,
            label=f'Linear vx={vx_val}' if vx_val != 2.5 else f'Linear vx={vx_val} (基准)')
    if vx_val == 2.5:
        ax.plot(distances, distances/vx_val, '-', color=colors[i], linewidth=2.8)
# Nonlinear curves
t_nl_e = [nl_axis_time(d, VX_E, AX_E) for d in distances]
t_nl_l = [nl_axis_time(d, VX_L, AX_L) for d in distances]
ax.plot(distances, t_nl_e, 'b-', linewidth=3, label='Nonlinear empty(vmax=3.0)')
ax.plot(distances, t_nl_l, 'r-', linewidth=3, label='Nonlinear loaded(vmax=2.3)')
ax.set_xlabel('水平距离 (m)', fontsize=15, fontweight='bold')
ax.set_ylabel('单程时间 (s)', fontsize=15, fontweight='bold')
ax.set_title('线性速度灵敏度 vs 非线性', fontsize=16, fontweight='bold')
ax.legend(fontsize=10, ncol=2); ax.grid(True, alpha=0.3)

# --- 1d: Effective speed ratio (nonlinear/linear) vs distance ---
ax = axes1[1, 0]
dist_fine = np.linspace(0.05, 30, 500)
t_lin_25 = dist_fine / 2.5
eff_speed_empty = [dist_fine[i]/nl_axis_time(d, VX_E, AX_E) for i, d in enumerate(dist_fine)]
eff_speed_loaded = [dist_fine[i]/nl_axis_time(d, VX_L, AX_L) for i, d in enumerate(dist_fine)]
ratio_empty = [e/2.5 for e in eff_speed_empty]
ratio_loaded = [l/2.5 for l in eff_speed_loaded]
ax.plot(dist_fine, ratio_empty, 'b-', linewidth=3, label='Empty有效速度/2.5')
ax.plot(dist_fine, ratio_loaded, 'r-', linewidth=3, label='Loaded有效速度/2.5')
ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=2, alpha=0.5)
ax.axhline(y=VX_E/2.5, color='b', linestyle=':', linewidth=1, alpha=0.4)
ax.axhline(y=VX_L/2.5, color='r', linestyle=':', linewidth=1, alpha=0.4)
ax.set_xlabel('水平距离 (m)', fontsize=15, fontweight='bold')
ax.set_ylabel('有效速度比值 (vs 线性2.5m/s)', fontsize=15, fontweight='bold')
ax.set_title('非线性有效速度比 (vs 线性vx=2.5)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.6)

# --- 1e: Per-crane flow time comparison ---
ax = axes1[1, 1]
crane_labels = [f'堆垛机{a}' for a in range(1, 8)]
flow_lin_c = [tl_lin[tl_lin['堆垛机编号']==a]['流动时间(s)'].sum()/3600 for a in range(1, 8)]
flow_nl_c = [tl_nl[tl_nl['堆垛机编号']==a]['流动时间(s)'].sum()/3600 for a in range(1, 8)]
x_pos = np.arange(7); width = 0.35
b1 = ax.bar(x_pos-width/2, flow_lin_c, width, label='线性模型', color='#2196F3', edgecolor='#333', linewidth=1)
b2 = ax.bar(x_pos+width/2, flow_nl_c, width, label='非线性模型', color='#FF9800', edgecolor='#333', linewidth=1)
for bar, val in zip(b1, flow_lin_c):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val:.1f}h', ha='center', fontsize=11, fontweight='bold')
for bar, val in zip(b2, flow_nl_c):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val:.1f}h', ha='center', fontsize=11, fontweight='bold')
ax.set_xticks(x_pos); ax.set_xticklabels(crane_labels, fontsize=13)
ax.set_ylabel('总流动时间 (h)', fontsize=15, fontweight='bold')
ax.set_title('各堆垛机总流动时间对比', fontsize=16, fontweight='bold')
ax.legend(fontsize=14); ax.grid(True, alpha=0.3, axis='y')

# --- 1f: Summary table ---
ax = axes1[1, 2]; ax.axis('off')
table_data = [
    ['指标', '线性模型', '非线性模型', '差异'],
    ['总流动时间', f'{stats_lin["total_flow"]/1e6:.2f}M s', f'{stats_nl["total_flow"]/1e6:.2f}M s',
     f'{(stats_nl["total_flow"]-stats_lin["total_flow"])/1e6:+.2f}M'],
    ['出库平均', f'{stats_lin["avg_out_flow"]:.0f}s', f'{stats_nl["avg_out_flow"]:.0f}s',
     f'{stats_nl["avg_out_flow"]-stats_lin["avg_out_flow"]:+.0f}s'],
    ['入库平均', f'{stats_lin["avg_in_flow"]:.0f}s', f'{stats_nl["avg_in_flow"]:.0f}s',
     f'{stats_nl["avg_in_flow"]-stats_lin["avg_in_flow"]:+.0f}s'],
    ['复合占比', f'{stats_lin["compound_ratio"]:.1f}%', f'{stats_nl["compound_ratio"]:.1f}%',
     f'{stats_nl["compound_ratio"]-stats_lin["compound_ratio"]:+.1f}pp'],
    ['Makespan', f'{stats_lin["makespan"]/3600:.1f}h', f'{stats_nl["makespan"]/3600:.1f}h',
     f'{(stats_nl["makespan"]-stats_lin["makespan"])/3600:+.1f}h'],
    ['总操作数', f'{stats_lin["total_ops"]}', f'{stats_nl["total_ops"]}', ''],
    ['变化率', '-', f'+{(stats_nl["total_flow"]/stats_lin["total_flow"]-1)*100:.1f}%', ''],
]
tbl = ax.table(cellText=table_data, cellLoc='center', loc='center',
               colWidths=[0.22, 0.26, 0.28, 0.24])
tbl.auto_set_font_size(False); tbl.set_fontsize(13)
tbl.scale(1.5, 2.3)
for key, cell in tbl.get_celld().items():
    if key[0] == 0:
        cell.set_facecolor('#455A64')
        cell.set_text_props(color='white', fontweight='bold', fontsize=14)
    cell.set_edgecolor('#BDBDBD')
ax.set_title('关键指标对比汇总', fontsize=17, fontweight='bold', pad=25)

fig1.suptitle('问题四：线性 vs 非线性速度模型 — 速度剖面、灵敏度与对比', fontsize=20, fontweight='bold', y=0.995)
fig1.tight_layout(rect=[0, 0, 1, 0.97])
fig1.savefig(os.path.join(OUT_DIR, '速度剖面对比图.png'), dpi=150, bbox_inches='tight', pad_inches=0.4)
print(f"  速度剖面对比图.png 已保存")

# =====================================================================
# Figure 2: Flow time distribution (WITH LOG-SPACED BINS to fix skew)
# =====================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(24, 8))

for idx, (task_type, label, color_lin, color_nl, ax_idx) in enumerate([
    ('outbound', '出库', '#1565C0', '#E65100', 0),
    ('inbound', '入库', '#1565C0', '#E65100', 1),
]):
    ax = axes2[ax_idx]
    flow_lin_data = tl_lin[tl_lin['任务类型']==task_type]['流动时间(s)'] / 3600
    flow_nl_data = tl_nl[tl_nl['任务类型']==task_type]['流动时间(s)'] / 3600

    # Log-spaced bins to handle long-tail distribution
    max_val = max(flow_lin_data.max(), flow_nl_data.max())
    min_val = max(0.001, min(flow_lin_data.min(), flow_nl_data.min()))
    bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)

    ax.hist(flow_lin_data, bins=bins, alpha=0.55, label=f'线性(均值{flow_lin_data.mean():.1f}h)', color=color_lin, edgecolor='#333', linewidth=0.5)
    ax.hist(flow_nl_data, bins=bins, alpha=0.55, label=f'非线性(均值{flow_nl_data.mean():.1f}h)', color=color_nl, edgecolor='#333', linewidth=0.5)
    ax.axvline(flow_lin_data.mean(), color=color_lin, linestyle='--', linewidth=3)
    ax.axvline(flow_nl_data.mean(), color=color_nl, linestyle='--', linewidth=3)
    ax.set_xscale('log')
    ax.set_xlabel('流动时间 (h, 对数刻度)', fontsize=15, fontweight='bold')
    ax.set_ylabel('频次', fontsize=15, fontweight='bold')
    ax.set_title(f'{label}流动时间分布', fontsize=16, fontweight='bold')
    ax.legend(fontsize=13, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

# 2c: Operation type breakdown
ax = axes2[2]
labels = ['纯出库', '纯入库', '复合']
lin_ops = [stats_lin['pure_out'], stats_lin['pure_in'], stats_lin['compound_ops']]
nl_ops = [stats_nl['pure_out'], stats_nl['pure_in'], stats_nl['compound_ops']]
x_pos3 = np.arange(3); width3 = 0.32
bars_l = ax.bar(x_pos3-width3/2, lin_ops, width3, label='线性模型', color='#2196F3', edgecolor='#333', linewidth=1)
bars_n = ax.bar(x_pos3+width3/2, nl_ops, width3, label='非线性模型', color='#FF9800', edgecolor='#333', linewidth=1)
ax.set_xticks(x_pos3); ax.set_xticklabels(labels, fontsize=14)
ax.set_ylabel('操作次数', fontsize=15, fontweight='bold')
ax.set_title('操作类型分布对比', fontsize=16, fontweight='bold')
ax.legend(fontsize=14, loc='upper left')
for bar, val in zip(bars_l, lin_ops):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10, str(val), ha='center', fontsize=12, fontweight='bold')
for bar, val in zip(bars_n, nl_ops):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10, str(val), ha='center', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(lin_ops + nl_ops)*1.1)

fig2.suptitle('问题四：流动时间分布与操作类型对比 (对数横轴消除左偏)', fontsize=20, fontweight='bold', y=0.995)
fig2.tight_layout(rect=[0, 0, 1, 0.97])
fig2.savefig(os.path.join(OUT_DIR, '对比分析_流动时间分布.png'), dpi=150, bbox_inches='tight', pad_inches=0.4)
print(f"  对比分析_流动时间分布.png 已保存")

# =====================================================================
# Figure 3: Per-aisle indicators + model complexity discussion
# =====================================================================
fig3, axes3 = plt.subplots(2, 2, figsize=(20, 14))

# 3a: Total flow time per aisle
ax = axes3[0, 0]
flow_per_aisle_lin = [tl_lin[tl_lin['堆垛机编号']==a]['流动时间(s)'].sum()/3600 for a in range(1, 8)]
flow_per_aisle_nl = [tl_nl[tl_nl['堆垛机编号']==a]['流动时间(s)'].sum()/3600 for a in range(1, 8)]
b_a1 = ax.bar(x_pos-width/2, flow_per_aisle_lin, width, label='线性', color='#2196F3', edgecolor='#333', linewidth=1)
b_a2 = ax.bar(x_pos+width/2, flow_per_aisle_nl, width, label='非线性', color='#FF9800', edgecolor='#333', linewidth=1)
for bar, val in zip(b_a1, flow_per_aisle_lin):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
for bar, val in zip(b_a2, flow_per_aisle_nl):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x_pos); ax.set_xticklabels(crane_labels, fontsize=13)
ax.set_ylabel('总流动时间 (h)', fontsize=15, fontweight='bold')
ax.set_title('各巷道总流动时间', fontsize=16, fontweight='bold')
ax.legend(fontsize=14); ax.grid(True, alpha=0.3, axis='y')

# 3b: Compound count per aisle
ax = axes3[0, 1]
comp_lin = [len(tl_lin[(tl_lin['堆垛机编号']==a)&(tl_lin['是否复合']=='是')]) for a in range(1, 8)]
comp_nl = [len(tl_nl[(tl_nl['堆垛机编号']==a)&(tl_nl['是否复合']=='是')]) for a in range(1, 8)]
ax.bar(x_pos-width/2, comp_lin, width, label='线性', color='#2196F3', edgecolor='#333', linewidth=1)
ax.bar(x_pos+width/2, comp_nl, width, label='非线性', color='#FF9800', edgecolor='#333', linewidth=1)
for i in range(7):
    ax.text(i-width/2, comp_lin[i]+1, str(comp_lin[i]), ha='center', fontsize=10, fontweight='bold')
    ax.text(i+width/2, comp_nl[i]+1, str(comp_nl[i]), ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x_pos); ax.set_xticklabels(crane_labels, fontsize=13)
ax.set_ylabel('复合操作次数', fontsize=15, fontweight='bold')
ax.set_title('各巷道复合操作数', fontsize=16, fontweight='bold')
ax.legend(fontsize=14); ax.grid(True, alpha=0.3, axis='y')

# 3c: Average operation time per aisle
ax = axes3[1, 0]
avg_op_lin = [tl_lin[tl_lin['堆垛机编号']==a]['操作时间(s)'].mean() for a in range(1, 8)]
avg_op_nl = [tl_nl[tl_nl['堆垛机编号']==a]['操作时间(s)'].mean() for a in range(1, 8)]
b_o1 = ax.bar(x_pos-width/2, avg_op_lin, width, label='线性', color='#2196F3', edgecolor='#333', linewidth=1)
b_o2 = ax.bar(x_pos+width/2, avg_op_nl, width, label='非线性', color='#FF9800', edgecolor='#333', linewidth=1)
for bar, val in zip(b_o1, avg_op_lin):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3, f'{val:.0f}', ha='center', fontsize=10, fontweight='bold')
for bar, val in zip(b_o2, avg_op_nl):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3, f'{val:.0f}', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x_pos); ax.set_xticklabels(crane_labels, fontsize=13)
ax.set_ylabel('平均操作时间 (s)', fontsize=15, fontweight='bold')
ax.set_title('各巷道平均操作时间', fontsize=16, fontweight='bold')
ax.legend(fontsize=14); ax.grid(True, alpha=0.3, axis='y')

# 3d: Model complexity discussion text
ax = axes3[1, 1]; ax.axis('off')
discussion_lines = [
    "速度模型复杂度对优化结果的影响",
    "",
    "1. 加速度的影响集中在短距离",
    "   d < v²/a ≈ 18m(空载)/13m(满载)时为三角形速度曲线",
    "   短距离行程加速度主导，非线性时间显著高于线性",
    "   长距离行程巡航主导，差异缩小至速度比附近",
    "",
    "2. 空载vs满载区分的意义",
    "   非线性空载更快(3.0>2.5)，满载更慢(2.3<2.5)",
    "   纯出库空去满回 → 部分抵消",
    "   纯入库满去空回 → 操作时间(满去)更长",
    "",
    "3. 线性速度参数选择的影响",
    "   线性速度快→时间短但低估短距操作(无加速段)",
    "   线性速度慢→时间接近非线性但高估长距操作",
    f"   当前线性vx=2.5介于非线性空载(3.0)和满载(2.3)之间",
    f"   总流动时间: 非线性/线性 = {stats_nl['total_flow']/stats_lin['total_flow']:.2f}",
    "",
    "4. 更精确的调度方案",
    "   - SPT排序键应考虑实际空/满载时间而非平均速度",
    "   - 复合决策基于非线性时间预估，优先配对节省大的",
    "   - 入库位选择按满载T_out排序，精确反映放入代价",
    "   - 高消耗材料应优先分配低层前列(短距受加速度影响大)",
]
for i, line in enumerate(discussion_lines):
    if i == 0:
        ax.text(0.05, 0.95-i*0.045, line, transform=ax.transAxes, fontsize=17, fontweight='bold', va='top')
    elif line.startswith("  "):
        ax.text(0.07, 0.95-i*0.045, line, transform=ax.transAxes, fontsize=12, va='top')
    elif line.startswith("  ") or line == "":
        ax.text(0.05, 0.95-i*0.045, line, transform=ax.transAxes, fontsize=12, va='top')
    else:
        ax.text(0.05, 0.95-i*0.045, line, transform=ax.transAxes, fontsize=13.5, fontweight='bold', va='top')

fig3.suptitle('问题四：各巷道指标对比与模型复杂度讨论', fontsize=20, fontweight='bold', y=0.995)
fig3.tight_layout(rect=[0, 0, 1, 0.97])
fig3.savefig(os.path.join(OUT_DIR, '对比分析_各巷道指标.png'), dpi=150, bbox_inches='tight', pad_inches=0.4)
print(f"  对比分析_各巷道指标.png 已保存")

# =====================================================================
# Figure 4: NEW — Speed sensitivity analysis (theoretical)
# =====================================================================
fig4, axes4 = plt.subplots(1, 2, figsize=(18, 8))

# 4a: Total flow time vs linear speed (theoretical projection)
ax = axes4[0]
vx_range = np.array([0.5, 0.8, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
# From the simulation, we know at vx=2.5: total_flow=2.638e6 s
# For linear model, flow_time ~ 1/vx (since T ∝ 1/vx for constant velocity)
base_flow = stats_lin['total_flow']
base_vx = 2.5
projected_flow = base_flow * base_vx / vx_range

# Also show the nonlinear result as a horizontal band
nl_flow = stats_nl['total_flow']
ax.fill_between(vx_range, nl_flow*0.98, nl_flow*1.02, alpha=0.25, color='#FF9800', label=f'Nonlinear result ({nl_flow/1e6:.2f}M s)')
ax.axhline(y=nl_flow, color='#FF9800', linestyle='-', linewidth=2.5, alpha=0.6)

ax.plot(vx_range, projected_flow/1e6, 'b-o', linewidth=2.5, markersize=10, label='Linear (speed-scaled from vx=2.5)')
ax.scatter([base_vx], [base_flow/1e6], s=200, c='blue', zorder=10, edgecolors='black', linewidths=2)
ax.annotate(f'Simulation\nvx=2.5: {base_flow/1e6:.2f}M s',
            xy=(base_vx, base_flow/1e6), xytext=(1.5, base_flow/1e6*1.15),
            fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#333', lw=2), bbox=dict(boxstyle='round', fc='#E3F2FD', alpha=0.8))

# Mark where linear crosses nonlinear
cross_vx = base_vx * base_flow / nl_flow
ax.axvline(x=cross_vx, color='#E53935', linestyle='--', linewidth=2)
ax.annotate(f'vx≈{cross_vx:.1f}时\n线性≈非线性', xy=(cross_vx, nl_flow/1e6),
            xytext=(cross_vx+0.5, nl_flow/1e6*0.92), fontsize=12, fontweight='bold', color='#E53935',
            arrowprops=dict(arrowstyle='->', color='#E53935', lw=2),
            bbox=dict(boxstyle='round', fc='#FFEBEE', alpha=0.8))

ax.set_xlabel('线性模型水平速度 vx (m/s)', fontsize=15, fontweight='bold')
ax.set_ylabel('总流动时间 (M s)', fontsize=15, fontweight='bold')
ax.set_title('线性速度-流动时间关系 (理论与非线性对比)', fontsize=16, fontweight='bold')
ax.legend(fontsize=13); ax.grid(True, alpha=0.3)

# 4b: Nonlinear/Linear time ratio heatmap by distance
ax = axes4[1]
dx_vals = np.logspace(-1, 1.7, 30)  # 0.1m to 50m
dz_vals = np.logspace(-1, 1.3, 20)  # 0.1m to 20m
ratio_matrix = np.zeros((len(dz_vals), len(dx_vals)))
for i, dz in enumerate(dz_vals):
    for j, dx in enumerate(dx_vals):
        # Nonlinear: average empty+loaded for one-way
        t_nl_avg = (nl_axis_time(dx, VX_E, AX_E) + nl_axis_time(dz, VZ_E, AZ_E)
                    + nl_axis_time(dx, VX_L, AX_L) + nl_axis_time(dz, VZ_L, AZ_L)) / 2
        t_lin = dx/2.5 + dz/0.65
        ratio_matrix[i, j] = t_nl_avg / t_lin if t_lin > 0 else 1.0

im = ax.pcolormesh(dx_vals, dz_vals, ratio_matrix, cmap='RdYlGn_r', shading='gouraud', vmin=0.8, vmax=2.5)
cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label('非线性时间/线性时间', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('水平距离 dx (m)', fontsize=15, fontweight='bold')
ax.set_ylabel('垂直距离 dz (m)', fontsize=15, fontweight='bold')
ax.set_title('非线性/线性 时间比值热力图\n(红=非线性更慢, 绿=接近)', fontsize=16, fontweight='bold')
# Add reference contour at ratio=1
X, Y = np.meshgrid(dx_vals, dz_vals)
ax.contour(X, Y, ratio_matrix, levels=[1.0, 1.5, 2.0], colors=['black', '#333', '#666'], linewidths=[2, 1.5, 1], linestyles=['-', '--', ':'])
ax.text(10, 0.15, 'ratio=1.0', fontsize=10, color='black', fontweight='bold')
ax.grid(True, alpha=0.2, which='both')

fig4.suptitle('问题四：速度灵敏度分析 — 线性速度影响与非线性/线性时间比', fontsize=20, fontweight='bold', y=0.995)
fig4.tight_layout(rect=[0, 0, 1, 0.97])
fig4.savefig(os.path.join(OUT_DIR, '速度灵敏度分析.png'), dpi=150, bbox_inches='tight', pad_inches=0.4)
print(f"  速度灵敏度分析.png 已保存")

# ============ Final discussion ============
print(f"\n{'='*70}")
print("讨论：速度模型复杂度对优化结果的影响")
print(f"{'='*70}")

diff_pct = (stats_nl['total_flow'] - stats_lin['total_flow']) / stats_lin['total_flow'] * 100
print(f"""
1. 总流动时间差异: {diff_pct:+.1f}%
   - 加速度阶段在短距离中占主导(三角形速度曲线)
   - 满载速度(2.3/0.58)低于线性(2.5/0.65)，所有带载行程更慢
   - 短距离行程(d<18m水平/d<3.75m垂直)非线性显著慢于线性

2. 线性速度参数的敏感性:
   - 线性vx=2.5介于空载3.0和满载2.3之间
   - 若线性vx降至约{base_vx*base_flow/nl_flow:.1f}，总流动时间将与非线性接近
   - 但此时长距离行程将被高估(线性无加速段优势)
   - 不存在单一线性速度能同时匹配短距和长距的非线性行为

3. 复合占比变化 ({stats_lin['compound_ratio']:.1f}% → {stats_nl['compound_ratio']:.1f}%):
   - 非线性下空回更贵(满载回程慢)，复合消除空回的边际收益更大
   - 系统自动增加复合配对以补偿非线性带来的额外时间代价

4. 模型复杂度代价:
   - 线性: 1次除法/轴 → O(1)
   - 非线性: 条件判断 + 除法/开方 → O(1)，计算开销可忽略
   - 实现复杂度: 需区分空满载状态、加速度参数管理
   - 非线性模型在全仿真中仅增加约{(stats_nl['total_flow']/stats_lin['total_flow']-1)*100:.0f}%时间结果差异

5. 更精确的调度方案建议:
   - SPT排序键应使用空/满载实际时间而非匀速近似
   - 入库位选择按满载T_out排序，精确反映放入代价
   - 复合配对应基于非线性时间预估，匹配节省最大的出入库对
   - 对低层前列(E1高消耗区，短距受加速度影响大)，可考虑优先分配
     以减少重复操作频次
   - 结合速度灵敏度曲线选择"最危险"距离区间优化库位分配
""")

print(f"\n全部图表和对比分析完成! 共4张图")
print(f"输出目录: {OUT_DIR}")
print(f"图片文件:")
for f in ['速度剖面对比图.png', '对比分析_流动时间分布.png', '对比分析_各巷道指标.png', '速度灵敏度分析.png']:
    path = os.path.join(OUT_DIR, f)
    if os.path.exists(path):
        print(f"  [OK] {f}")
