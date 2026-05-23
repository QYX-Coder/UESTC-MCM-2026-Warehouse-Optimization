"""
问题一 出库时间推导 (匀速直线, 顺序运动)
T = x*w/v_xe + H(z)/v_ze + x*w/v_xl + H(z)/v_zl + t_fork + t_other
"""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE, '输出结果', '问题一')
CSV_DIR = os.path.join(OUT_DIR, '分配结果CSV')

# Physics
COL_W = 0.4  # m
LH = {}; cum = 0.0
for z in range(1, 51):
    if z <= 8: h = 0.2
    elif z <= 42: h = 0.4
    else: h = 0.5
    LH[z] = cum + h / 2; cum += h

VXE, VZE = 3.0, 0.75  # empty
VXL, VZL = 2.3, 0.58  # loaded
TF = 3.0; TO = 3.0     # fork + other

df = pd.read_csv(os.path.join(CSV_DIR, '完整分配明细.csv'))
x, z = df['列(x)'].values, df['层(z)'].values
Dx, Dz = x * COL_W, np.array([LH[zi] for zi in z])

T_get  = Dx/VXE + Dz/VZE
T_back = Dx/VXL + Dz/VZL
T = T_get + T_back + TF + TO

w = df['消耗占比'] / df['消耗占比'].sum()
w_mean = np.average(T, weights=w)
a_mean = T.mean()

print("=" * 60)
print("Average Retrieval Time (constant speed, sequential)")
print("=" * 60)
print(f"""
Formula:
  T_get  = x*{COL_W}/v_xe + H(z)/v_ze  (empty, sequential)
  T_back = x*{COL_W}/v_xl + H(z)/v_zl  (loaded, sequential)
  T = T_get + T_back + t_fork({TF}s) + t_other({TO}s)

  Arithmetic mean:  {a_mean:.2f} s
  Weighted mean:    {w_mean:.2f} s
  Min: {T.min():.1f}s  Max: {T.max():.1f}s  Median: {np.median(T):.1f}s

  H component (empty):  mean={Dz.mean()/VZE:.1f}s
  X component (empty):  mean={Dx.mean()/VXE:.1f}s
""")

# Save
td = df[['库位编号', '原材料编号', '箱子类型', '列(x)', '层(z)']].copy()
td['取货时间(s)'] = T_get; td['送货时间(s)'] = T_back
td['单次出库时间(s)'] = T
td.to_csv(os.path.join(CSV_DIR, '出库时间明细.csv'), index=False, encoding='utf-8-sig')
print(f"Saved: {os.path.join(CSV_DIR, '出库时间明细.csv')}")
