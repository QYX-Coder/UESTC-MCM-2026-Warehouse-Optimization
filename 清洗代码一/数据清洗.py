"""
数据清洗脚本 - 库位分配优化问题
对三张原始数据表进行清洗和预处理
"""
import pandas as pd
import numpy as np
import os

# ============ 路径配置 ============
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '数据集一')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '清洗数据一')

# ============ 1. 读取原始数据 ============
print("=" * 50)
print("读取原始数据...")

df1 = pd.read_csv(os.path.join(DATA_DIR, '表1_原材料库存数据.csv'))
df2 = pd.read_csv(os.path.join(DATA_DIR, '表2_生产线订料数据.csv'))
df3 = pd.read_csv(os.path.join(DATA_DIR, '表3_入库材料数据.csv'))

print(f"表1 原材料库存数据: {df1.shape[0]} 行, {df1.shape[1]} 列")
print(f"表2 生产线订料数据: {df2.shape[0]} 行, {df2.shape[1]} 列")
print(f"表3 入库材料数据: {df3.shape[0]} 行, {df3.shape[1]} 列")

# ============ 2. 列名清洗 ============
# 去除列名首尾空格
for df in [df1, df2, df3]:
    df.columns = df.columns.str.strip()

print("\n表1 列名:", df1.columns.tolist())
print("表2 列名:", df2.columns.tolist())
print("表3 列名:", df3.columns.tolist())

# ============ 3. 表1 清洗 ============
print("\n" + "=" * 50)
print("清洗表1 - 原材料库存数据")

# 去尾随逗号产生的空列
df1 = df1.loc[:, ~df1.columns.str.startswith('Unnamed')]

# 检查缺失值
print(f"缺失值统计:\n{df1.isnull().sum()}")

# 检查重复
print(f"重复行数: {df1.duplicated().sum()}")

# 数据类型转换
df1['原材料编号'] = df1['原材料编号'].astype(str).str.strip()
df1['箱子类型'] = df1['箱子类型'].astype(str).str.strip()
df1['消耗占比'] = pd.to_numeric(df1['消耗占比'], errors='coerce')
df1['库存数量/箱'] = pd.to_numeric(df1['库存数量/箱'], errors='coerce')

# 去除可能存在的空行
df1 = df1.dropna(subset=['原材料编号'])
df1 = df1[df1['原材料编号'] != '']

# 统计数据
print(f"清洗后行数: {df1.shape[0]}")
print(f"箱子类型分布:\n{df1['箱子类型'].value_counts()}")
print(f"消耗占比之和: {df1['消耗占比'].sum():.4f}")
print(f"库存总量(箱): {df1['库存数量/箱'].sum()}")

# 找出消耗占比异常值（如 > 1 的值，这些可能是百分比而非小数，但题目说和为1）
# 检查是否有异常大的值
large_ratio = df1[df1['消耗占比'] > 1]
if len(large_ratio) > 0:
    print(f"\n消耗占比>1的材料数: {len(large_ratio)}")
    print(large_ratio[['原材料编号', '箱子类型', '消耗占比']].head(10))

# ============ 4. 表2 清洗 ============
print("\n" + "=" * 50)
print("清洗表2 - 生产线订料数据")

df2 = df2.loc[:, ~df2.columns.str.startswith('Unnamed')]
print(f"缺失值统计:\n{df2.isnull().sum()}")

df2['订单号'] = pd.to_numeric(df2['订单号'], errors='coerce')
df2['箱号'] = df2['箱号'].astype(str).str.strip()
df2['箱子类型'] = df2['箱子类型'].astype(str).str.strip()
df2['原材料号'] = df2['原材料号'].astype(str).str.strip()
df2['单位'] = df2['单位'].astype(str).str.strip()
df2['时间'] = pd.to_datetime(df2['时间'], errors='coerce')

df2 = df2.dropna(subset=['订单号', '原材料号'])
print(f"清洗后行数: {df2.shape[0]}")
print(f"时间范围: {df2['时间'].min()} ~ {df2['时间'].max()}")
print(f"涉及原材料数: {df2['原材料号'].nunique()}")
print(f"箱子类型分布:\n{df2['箱子类型'].value_counts()}")

# ============ 5. 表3 清洗 ============
print("\n" + "=" * 50)
print("清洗表3 - 入库材料数据")

df3 = df3.loc[:, ~df3.columns.str.startswith('Unnamed')]
print(f"缺失值统计:\n{df3.isnull().sum()}")

df3['入库批次'] = df3['入库批次'].astype(str).str.strip()
df3['原材料编号'] = df3['原材料编号'].astype(str).str.strip()
df3['箱子类型'] = df3['箱子类型'].astype(str).str.strip()
df3['入库数量/箱'] = pd.to_numeric(df3['入库数量/箱'], errors='coerce')
df3['入库时间'] = pd.to_datetime(df3['入库时间'], errors='coerce')

df3 = df3.dropna(subset=['入库批次', '原材料编号'])
print(f"清洗后行数: {df3.shape[0]}")
print(f"入库批次总数: {df3['入库批次'].nunique()}")
print(f"入库总箱数: {df3['入库数量/箱'].sum()}")
print(f"时间范围: {df3['入库时间'].min()} ~ {df3['入库时间'].max()}")
print(f"箱子类型分布:\n{df3['箱子类型'].value_counts()}")

# ============ 6. 保存清洗后数据 ============
print("\n" + "=" * 50)
print("保存清洗后数据...")

df1.to_csv(os.path.join(OUT_DIR, '清洗后_原材料库存数据.csv'), index=False, encoding='utf-8-sig')
df2.to_csv(os.path.join(OUT_DIR, '清洗后_生产线订料数据.csv'), index=False, encoding='utf-8-sig')
df3.to_csv(os.path.join(OUT_DIR, '清洗后_入库材料数据.csv'), index=False, encoding='utf-8-sig')

# 同时保存Excel格式
with pd.ExcelWriter(os.path.join(OUT_DIR, '清洗后_全部数据.xlsx'), engine='openpyxl') as writer:
    df1.to_excel(writer, sheet_name='原材料库存数据', index=False)
    df2.to_excel(writer, sheet_name='生产线订料数据', index=False)
    df3.to_excel(writer, sheet_name='入库材料数据', index=False)

# ============ 7. 生成数据摘要 ============
print("\n生成数据摘要...")
summary_lines = []
summary_lines.append(f"## 数据清洗摘要\n")
summary_lines.append(f"- 清洗日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
summary_lines.append(f"- 表1 原材料库存数据: {df1.shape[0]} 条记录, {df1['原材料编号'].nunique()} 种原材料")
summary_lines.append(f"  - E1箱: {(df1['箱子类型']=='E1').sum()} 种, E3箱: {(df1['箱子类型']=='E3').sum()} 种, E4箱: {(df1['箱子类型']=='E4').sum()} 种")
summary_lines.append(f"  - 消耗占比总和: {df1['消耗占比'].sum():.4f}")
summary_lines.append(f"  - 库存总量: {df1['库存数量/箱'].sum()} 箱")
summary_lines.append(f"- 表2 生产线订料数据: {df2.shape[0]} 条记录")
summary_lines.append(f"  - 时间范围: {df2['时间'].min()} ~ {df2['时间'].max()}")
summary_lines.append(f"  - 涉及 {df2['原材料号'].nunique()} 种原材料")
summary_lines.append(f"- 表3 入库材料数据: {df3.shape[0]} 条记录")
summary_lines.append(f"  - 入库总箱数: {df3['入库数量/箱'].sum()} 箱")
summary_lines.append(f"  - 时间范围: {df3['入库时间'].min()} ~ {df3['入库时间'].max()}")

with open(os.path.join(OUT_DIR, '数据清洗说明.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))

print("数据清洗完成!")
print(f"清洗代码: {os.path.join(os.path.dirname(__file__), '数据清洗.py')}")
print(f"清洗数据: {OUT_DIR}")
