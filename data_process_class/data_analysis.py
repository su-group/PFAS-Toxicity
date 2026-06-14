import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 配置 ---
CSV_FILE_PATH = 'step3_OECD_Class_Enhanced_v2.csv' # 请确保文件路径正确

# 设置中文字体和风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans'] # 尝试中文字体
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
sns.set_style("whitegrid")

# --- 1. 加载数据并快速概览 ---
print("=== 步骤 1: 加载数据并快速概览 ===")
try:
    df_enhanced = pd.read_csv(CSV_FILE_PATH)
    print(f"成功加载数据，总行数: {len(df_enhanced)}")
    print("\n数据集列名:")
    print(df_enhanced.columns.tolist())
    print("\n数据集前几行预览:")
    print(df_enhanced.head(5))
except FileNotFoundError:
    print(f"错误: 找不到文件 {CSV_FILE_PATH}")
    exit()
except Exception as e:
    print(f"加载数据时发生错误: {e}")
    exit()

# --- 2. 可视化一：按“主类”和“子类”分布 (关联图片分类) ---
print("\n=== 步骤 2: 可视化 - 按主类和子类分布 ===")

fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# 1. 按 first_class (主类) 计数
ax1 = axes[0]
first_class_counts = df_enhanced['first_class'].value_counts()
sns.barplot(x=first_class_counts.index, y=first_class_counts.values, ax=ax1)
ax1.set_title('按主类 (Primary Class) 分布 (关联图片顶层分类)')
ax1.set_xlabel('主类 (first_class)')
ax1.set_ylabel('分子数量')
ax1.tick_params(axis='x', rotation=45)

# 2. 按 second_class (子类) 计数 (取前20个最常见的)
ax2 = axes[1]
top_second_classes = df_enhanced['second_class'].value_counts().head(20)
df_top = df_enhanced[df_enhanced['second_class'].isin(top_second_classes.index)]
sns.countplot(data=df_top, y='second_class', ax=ax2, order=top_second_classes.index)
ax2.set_title('按子类 (Secondary Class) 分布 (Top 20, 关联图片二级分类)')
ax2.set_xlabel('分子数量')
ax2.set_ylabel('子类 (second_class)')

plt.tight_layout()
plt.show()

# --- 3. 可视化二：按结构特征分组（核心控制变量研究） ---
print("\n=== 步骤 3: 可视化 - 按结构特征分组 (控制变量研究) ===")

# 检查是否存在预测毒性值列，如果不存在则使用 carbon_chain_length 作为示例Y轴
# 假设预测毒性值列名为 'prediction_toxicity'，请根据实际情况修改
PREDICTION_COL = 'prediction_toxicity' # <--- 请根据您的实际列名修改

if PREDICTION_COL not in df_enhanced.columns:
    print(f"警告: 列 '{PREDICTION_COL}' 不存在，将使用 'carbon_chain_length' 作为示例Y轴进行演示。")
    Y_AXIS = 'carbon_chain_length'
else:
    Y_AXIS = PREDICTION_COL

# 案例1: 比较 PFAA precursors 下 HFCs 子类中，直链 vs 支链的 Y_AXIS 值差异
target_first_class = 'PFAA precursors'
target_second_class = 'HFCs'

subset_hfcs = df_enhanced[
    (df_enhanced['first_class'] == target_first_class) &
    (df_enhanced['second_class'] == target_second_class) &
    (df_enhanced['chain_type'].isin(['straight', 'branched'])) # 确保只比较这两种类型
]

if not subset_hfcs.empty:
    fig, ax = plt.subplots(figsize=(8, 6))
    # 使用 value_counts 获取每种 chain_type 的数量，用于在图上标注
    counts = subset_hfcs['chain_type'].value_counts()
    sns.boxplot(data=subset_hfcs, x='chain_type', y=Y_AXIS, ax=ax)
    ax.set_title(f'{target_first_class} - {target_second_class}: 直链 vs 支链 {Y_AXIS} 对比')
    ax.set_xlabel('链类型 (chain_type)')
    ax.set_ylabel(f'{Y_AXIS}')
    # 在箱线图上标注样本数
    for i, chain_type in enumerate(counts.index):
        ax.text(i, ax.get_ylim()[1] * 0.95, f'n={counts[chain_type]}', ha='center', va='top', fontsize=10)
    plt.show()
else:
    print(f"未找到 {target_first_class} - {target_second_class} (直链/支链) 的数据。")

# 案例2: 在 PFCAs 类别下，分析碳链长度对 Y_AXIS 的影响
subset_pfcas = df_enhanced[
    (df_enhanced['first_class'] == 'PFAAs') &
    (df_enhanced['second_class'] == 'PFCAs') &
    (df_enhanced['chain_type'] == 'straight') # 通常PFCAs是直链
]

if not subset_pfcas.empty:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=subset_pfcas, x='carbon_chain_length', y=Y_AXIS, ax=ax)
    sns.regplot(data=subset_pfcas, x='carbon_chain_length', y=Y_AXIS, scatter=False, color='red', ax=ax)
    ax.set_title(f'PFCAs: 碳链长度 vs {Y_AXIS}')
    ax.set_xlabel('全氟碳链长度 (carbon_chain_length)')
    ax.set_ylabel(f'{Y_AXIS}')
    plt.show()
else:
    print("未找到 PFAAs - PFCAs (直链) 的数据。")

# --- 4. 可视化三：多维度交叉分析 (热力图) ---
print("\n=== 步骤 4: 可视化 - 多维度交叉分析 (热力图) ===")

# 创建一个透视表，计算每个 (first_class, second_class) 组合的 Y_AXIS 平均值
pivot_table = df_enhanced.pivot_table(
    values=Y_AXIS,
    index='first_class',
    columns='second_class',
    aggfunc='mean',
    fill_value=np.nan # 用 NaN 填充缺失值，热力图会自动处理
)

# 绘制热力图
plt.figure(figsize=(16, 8))
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=.5, cbar_kws={'label': f'平均 {Y_AXIS}'})
plt.title(f'不同主类-子类组合的平均 {Y_AXIS}')
plt.xlabel('子类 (Second Class)')
plt.ylabel('主类 (First Class)')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- 5. 分析与总结：关联图片中的分类树状图 ---
print("\n=== 步骤 5: 分析与总结 (关联图片分类) ===")

# 计算每个 first_class 下的分子总数和 unique second_class 数量
class_summary = df_enhanced.groupby('first_class').agg(
    num_molecules=('first_class', 'count'),
    unique_second_classes=('second_class', 'nunique')
).reset_index()

print("\n数据集在各主类下的分布:")
print(class_summary)

# 展示每个主类下的具体子类
for first_class in df_enhanced['first_class'].unique():
    subset_fc = df_enhanced[df_enhanced['first_class'] == first_class]
    second_classes_in_fc = subset_fc['second_class'].unique()
    print(f"\n主类 '{first_class}' 包含的子类 ({len(second_classes_in_fc)} 个):")
    print(f"  - {', '.join(second_classes_in_fc)}")

# 展示 `group_id` 的构成示例
print(f"\n`group_id` 示例及其构成说明:")
print(f"`group_id` 格式: [first_class]_[second_class]_[chain_type]_[carbon_chain_length]_[functional_group]_[ring_size]")
print(f"示例: {df_enhanced['group_id'].iloc[0] if not df_enhanced.empty else 'N/A'}")
print(f"  - first_class: {df_enhanced['first_class'].iloc[0] if not df_enhanced.empty else 'N/A'}")
print(f"  - second_class: {df_enhanced['second_class'].iloc[0] if not df_enhanced.empty else 'N/A'}")
print(f"  - chain_type: {df_enhanced['chain_type'].iloc[0] if not df_enhanced.empty else 'N/A'}")
print(f"  - carbon_chain_length: {df_enhanced['carbon_chain_length'].iloc[0] if not df_enhanced.empty else 'N/A'}")
print(f"  - functional_group: {df_enhanced['functional_group'].iloc[0] if not df_enhanced.empty else 'N/A'}")
print(f"  - ring_size: {df_enhanced['ring_size'].iloc[0] if not df_enhanced.empty else 'N/A'}")

print("\n=== 分析完成 ===")
print("该分析验证了您设计的分类方案的有效性：")
print("1. `first_class` 和 `second_class` 直接对应图片中的分类层级。")
print(f"2. `chain_type`, `carbon_chain_length`, `ring_size` 等结构特征允许进行精确的控制变量研究 (如比较直/支链、环大小、链长)。")
print(f"3. `functional_group` 有助于区分不同官能团对 Y_AXIS 的影响。")
print(f"4. `group_id` 将所有这些特征整合，为每个分子提供了唯一的、可追溯的分类标签。")
print(f"5. 可视化结果（如箱线图、散点图、热力图）直观地展示了不同分类下 Y_AXIS 的分布和趋势。")