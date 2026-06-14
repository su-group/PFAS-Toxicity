# 保存为单独的Python文件并运行所有图形
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os

# 设置中文字体和样式
# plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.style.use('seaborn-whitegrid')


# ==================== 数据读取部分 ====================
def load_and_merge_data(toxicity_file, labels_file, smiles_col='SMILES',
                        toxicity_col='predicted_toxicity', chain_type_col='chain_type'):
    """
    从两个CSV文件加载数据并合并

    参数:
    toxicity_file: 包含毒性和SMILES的CSV文件路径
    labels_file: 包含SMILES和支链标签的CSV文件路径
    smiles_col: SMILES列名
    toxicity_col: 毒性值列名
    chain_type_col: 支链类型列名
    """
    print("正在加载数据...")

    # 读取毒性数据文件
    print(f"  1. 读取毒性数据文件: {toxicity_file}")
    toxicity_df = pd.read_csv(toxicity_file)
    print(f"    读取到 {len(toxicity_df)} 条记录")

    # 检查必要的列是否存在
    if smiles_col not in toxicity_df.columns:
        raise ValueError(f"毒性数据文件缺少'{smiles_col}'列")

    if toxicity_col not in toxicity_df.columns:
        # 尝试查找可能的毒性列名
        possible_tox_cols = ['toxicity', 'predicted_toxicity', 'pLD50', 'LD50', 'value']
        for col in possible_tox_cols:
            if col in toxicity_df.columns:
                toxicity_col = col
                print(f"    使用 '{toxicity_col}' 作为毒性列")
                break
        else:
            raise ValueError(f"毒性数据文件缺少毒性列，请指定正确的列名")

    # 读取标签文件
    print(f"  2. 读取支链标签文件: {labels_file}")
    labels_df = pd.read_csv(labels_file)
    print(f"    读取到 {len(labels_df)} 条记录")

    if smiles_col not in labels_df.columns:
        raise ValueError(f"标签文件缺少'{smiles_col}'列")

    if chain_type_col not in labels_df.columns:
        # 尝试查找可能的支链类型列名
        possible_chain_cols = ['chain_type', 'branch_type', 'type', 'structure_type']
        for col in possible_chain_cols:
            if col in labels_df.columns:
                chain_type_col = col
                print(f"    使用 '{chain_type_col}' 作为支链类型列")
                break
        else:
            raise ValueError(f"标签文件缺少支链类型列，请指定正确的列名")

    # 合并数据
    print(f"  3. 合并数据 (基于 '{smiles_col}' 列)")
    merged_df = pd.merge(toxicity_df[[smiles_col, toxicity_col]],
                         labels_df[[smiles_col, chain_type_col]],
                         on=smiles_col, how='inner')

    print(f"    合并后得到 {len(merged_df)} 条记录")

    # 检查是否有缺失值
    missing_tox = merged_df[toxicity_col].isna().sum()
    missing_chain = merged_df[chain_type_col].isna().sum()

    if missing_tox > 0:
        print(f"    警告: 有 {missing_tox} 条记录的毒性值为空")
    if missing_chain > 0:
        print(f"    警告: 有 {missing_chain} 条记录的支链类型为空")

    # 移除缺失值
    merged_df = merged_df.dropna(subset=[toxicity_col, chain_type_col])
    print(f"    移除缺失值后: {len(merged_df)} 条记录")

    # 标准化支链类型标签
    # 将各种可能的直链标签统一为'straight'，支链标签统一为'branched'
    merged_df[chain_type_col] = merged_df[chain_type_col].str.lower().str.strip()

    # 定义可能的标签映射
    straight_labels = ['straight', 'linear', '直链', '线性']
    branched_labels = ['branched', 'branch', '支链', 'branched_chain']

    def standardize_chain_type(label):
        label_lower = str(label).lower()
        for s in straight_labels:
            if s in label_lower:
                return 'straight'
        for b in branched_labels:
            if b in label_lower:
                return 'branched'
        return label  # 如果无法识别，返回原值

    merged_df[chain_type_col] = merged_df[chain_type_col].apply(standardize_chain_type)

    # 统计各类别的数量
    chain_counts = merged_df[chain_type_col].value_counts()
    print(f"    类别分布: {dict(chain_counts)}")

    return merged_df, toxicity_col, chain_type_col


def prepare_plot_data(merged_df, toxicity_col, chain_type_col):
    """
    从合并的数据中准备绘图数据

    返回:
    straight_data: 直链数据数组
    branched_data: 支链数据数组
    """
    # 分离直链和支链数据
    straight_mask = merged_df[chain_type_col] == 'straight'
    branched_mask = merged_df[chain_type_col] == 'branched'

    straight_data = merged_df[straight_mask][toxicity_col].values
    branched_data = merged_df[branched_mask][toxicity_col].values

    print(f"\n数据准备完成:")
    print(f"  直链样本数: {len(straight_data)}")
    print(f"  支链样本数: {len(branched_data)}")

    if len(straight_data) == 0 or len(branched_data) == 0:
        raise ValueError("直链或支链数据为空，请检查数据文件")

    # 计算基本统计量
    straight_mean = np.mean(straight_data)
    branched_mean = np.mean(branched_data)
    straight_median = np.median(straight_data)
    branched_median = np.median(branched_data)

    print(f"  直链均值: {straight_mean:.3f}, 中位数: {straight_median:.3f}")
    print(f"  支链均值: {branched_mean:.3f}, 中位数: {branched_median:.3f}")

    # 进行t检验
    if len(straight_data) > 1 and len(branched_data) > 1:
        t_stat, p_value = stats.ttest_ind(straight_data, branched_data)
        print(f"  t检验: t = {t_stat:.3f}, p = {p_value:.4f}")
    else:
        t_stat, p_value = None, None
        print("  警告: 样本量过小，无法进行t检验")

    return straight_data, branched_data, straight_mean, branched_mean, straight_median, branched_median, t_stat, p_value


# ==================== 绘图函数 ====================
def create_box_plot(straight_data, branched_data, p_value, output_dir='toxicity_plots'):
    """创建箱线图"""
    fig, ax = plt.subplots(figsize=(10, 7))

    data = [straight_data, branched_data]
    labels = ['Straight', 'Branched']
    colors = ['skyblue', 'lightcoral']

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6, showmeans=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 添加数据点
    for i, (group_data, color) in enumerate(zip(data, colors)):
        x_pos = np.random.normal(i + 1, 0.04, size=len(group_data))
        ax.scatter(x_pos, group_data, alpha=0.4, color=color, s=30,
                   edgecolors='black', linewidth=0.5)

    # 显著性标记
    if p_value is not None:
        if p_value < 0.001:
            sig_symbol = '***'
        elif p_value < 0.01:
            sig_symbol = '**'
        elif p_value < 0.05:
            sig_symbol = '*'
        else:
            sig_symbol = 'ns'

        y_max = max(straight_data.max(), branched_data.max())
        y_min = min(straight_data.min(), branched_data.min())
        y_range = y_max - y_min

        if sig_symbol != 'ns':
            ax.plot([1, 1, 2, 2],
                    [y_max + 0.1 * y_range, y_max + 0.15 * y_range,
                     y_max + 0.15 * y_range, y_max + 0.1 * y_range],
                    lw=1.5, c='black')
            ax.text(1.5, y_max + 0.2 * y_range, sig_symbol,
                    ha='center', va='bottom', fontsize=18, fontweight='bold')

    ax.set_ylabel('LD₅₀(-log(mol/kg)', fontsize=18)
    ax.set_title('Toxicity Comparison: Box Plot', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 添加样本数量信息
    ax.text(0.02, 0.98, f'n(Straight) = {len(straight_data)}\nn(Branched) = {len(branched_data)}',
            transform=ax.transAxes, fontsize=18, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # 在create_box_plot和create_violin_plot函数中
    ax.tick_params(axis='x', labelsize=20, labelcolor='black')
    ax.tick_params(axis='y', labelsize=20, labelcolor='black')
    ax.set_xticklabels(['Straight', 'Branched'], fontsize=20)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/box_plot.png', dpi=500, bbox_inches='tight')
    plt.show()


def create_violin_plot(straight_data, branched_data, straight_mean, branched_mean, output_dir='toxicity_plots'):
    """创建小提琴图"""
    fig, ax = plt.subplots(figsize=(10, 7))

    data_for_violin = pd.DataFrame({
        'Toxicity': np.concatenate([straight_data, branched_data]),
        'Chain Type': ['Straight'] * len(straight_data) + ['Branched'] * len(branched_data)
    })

    colors = ['skyblue', 'lightcoral']
    sns.violinplot(x='Chain Type', y='Toxicity', data=data_for_violin,
                   ax=ax, palette=colors, inner='quartile', cut=0)

    # 添加均值标记
    ax.scatter(0, straight_mean, color='black', s=150, marker='^',
               edgecolor='white', linewidth=2, zorder=10)
    ax.scatter(1, branched_mean, color='black', s=150, marker='^',
               edgecolor='white', linewidth=2, zorder=10)

    ax.set_ylabel('LD₅₀(-log(mol/kg)', fontsize=18)
    ax.set_xlabel('Chain Type', fontsize=18)
    ax.set_title('Toxicity Comparison: Violin Plot', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 添加统计信息
    ax.text(0.02, 0.98, f'Straight Mean = {straight_mean:.3f}\nBranched Mean = {branched_mean:.3f}',
            transform=ax.transAxes, fontsize=18, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.tick_params(axis='x', labelsize=20, labelcolor='black')
    ax.tick_params(axis='y', labelsize=20, labelcolor='black')
    ax.set_xticklabels(['Straight', 'Branched'], fontsize=18)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/violin_plot.png', dpi=500, bbox_inches='tight')
    plt.show()


def create_histogram(straight_data, branched_data, output_dir='toxicity_plots'):
    """创建直方图"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # 确定合适的范围
    all_data = np.concatenate([straight_data, branched_data])
    data_range = (np.min(all_data) - 0.5, np.max(all_data) + 0.5)
    n_bins = min(30, int(len(all_data) / 10))  # 动态确定bin数量

    ax.hist(straight_data, bins=n_bins, range=data_range, alpha=0.6,
            color='skyblue', edgecolor='black', density=True, label='Straight')
    ax.hist(branched_data, bins=n_bins, range=data_range, alpha=0.6,
            color='lightcoral', edgecolor='black', density=True, label='Branched')

    # 添加KDE曲线
    kde_straight = gaussian_kde(straight_data)
    kde_branched = gaussian_kde(branched_data)
    x_range = np.linspace(data_range[0], data_range[1], 1000)

    ax.plot(x_range, kde_straight(x_range), color='darkblue', linewidth=2.5, label='Straight KDE')
    ax.plot(x_range, kde_branched(x_range), color='darkred', linewidth=2.5, label='Branched KDE')

    ax.set_xlabel('LD₅₀(-log(mol/kg)', fontsize=18)
    ax.set_ylabel('Probability Density', fontsize=18)
    ax.set_title('Toxicity Distribution Histogram', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=18)
    ax.tick_params(axis='x', labelsize=20, labelcolor='black')
    ax.tick_params(axis='y', labelsize=20, labelcolor='black')

    # 计算Jensen-Shannon散度
    hist_straight, bins = np.histogram(straight_data, bins=n_bins, range=data_range, density=True)
    hist_branched, _ = np.histogram(branched_data, bins=n_bins, range=data_range, density=True)
    # js_divergence = jensenshannon(hist_straight, hist_branched)
    #
    # ax.text(0.02, 0.98, f'JS Divergence = {js_divergence:.3f}',
    #         transform=ax.transAxes, fontsize=18, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/histogram.png', dpi=500, bbox_inches='tight')
    plt.show()


def create_cdf_plot(straight_data, branched_data, straight_median, branched_median, output_dir='toxicity_plots'):
    """创建CDF图"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # 计算CDF
    straight_sorted = np.sort(straight_data)
    branched_sorted = np.sort(branched_data)

    straight_cdf = np.arange(1, len(straight_sorted) + 1) / len(straight_sorted)
    branched_cdf = np.arange(1, len(branched_sorted) + 1) / len(branched_sorted)

    ax.plot(straight_sorted, straight_cdf, color='darkblue', linewidth=2.5, label='Straight')
    ax.plot(branched_sorted, branched_cdf, color='darkred', linewidth=2.5, label='Branched')

    # 添加中位数标记
    ax.axvline(straight_median, color='darkblue', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(branched_median, color='darkred', linestyle=':', linewidth=1.5, alpha=0.7)

    # 标记中位数点
    straight_median_cdf = np.interp(straight_median, straight_sorted, straight_cdf)
    branched_median_cdf = np.interp(branched_median, branched_sorted, branched_cdf)

    ax.scatter(straight_median, straight_median_cdf, color='darkblue', s=100,
               zorder=5, marker='o', edgecolor='white', linewidth=2, label='Straight Median')
    ax.scatter(branched_median, branched_median_cdf, color='darkred', s=100,
               zorder=5, marker='o', edgecolor='white', linewidth=2, label='Branched Median')
    #
    # # Kolmogorov-Smirnov检验
    # ks_stat, ks_pvalue = stats.ks_2samp(straight_data, branched_data)
    #
    # # 添加统计信息
    # stats_text = f'KS Test: D = {ks_stat:.3f}\n'
    # if ks_pvalue < 0.001:
    #     stats_text += 'p < 0.001 ***'
    # elif ks_pvalue < 0.01:
    #     stats_text += f'p = {ks_pvalue:.3f} **'
    # elif ks_pvalue < 0.05:
    #     stats_text += f'p = {ks_pvalue:.3f} *'
    # else:
    #     stats_text += f'p = {ks_pvalue:.3f} (ns)'
    #
    # ax.text(0.02, 0.98, stats_text,
    #         transform=ax.transAxes, fontsize=18, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('LD₅₀(-log(mol/kg)', fontsize=18)
    ax.set_ylabel('Cumulative Probability', fontsize=18)
    ax.set_title('Toxicity Cumulative Distribution Function', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=18)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', labelsize=20, labelcolor='black')
    ax.tick_params(axis='y', labelsize=20, labelcolor='black')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/cdf_plot.png', dpi=500, bbox_inches='tight')
    plt.show()


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("毒性比较可视化工具")
    print("=" * 60)

    # 配置您的文件路径和列名
    # 请根据您的实际情况修改这些路径和列名

    # 毒性数据文件路径（包含SMILES和毒性值）
    toxicity_file = "F:\GNN-pro\data_process_class\pfas_sar_analysis_predictions.csv"  # 请修改为您的文件路径

    # 支链标签文件路径（包含SMILES和支链类型）
    labels_file = "F:\GNN-pro\data_process_class\step3_OECD_Class_Enhanced_v2_with_CF.csv"  # 请修改为您的文件路径

    # 列名配置（根据您的CSV文件中的列名修改）
    smiles_col = "SMILES"  # SMILES列名
    toxicity_col = "predicted_toxicity"  # 毒性值列名
    chain_type_col = "chain_type"  # 支链类型列名

    # 输出目录
    output_dir = "toxicity_plots"

    try:
        # 1. 加载和合并数据
        merged_df, toxicity_col, chain_type_col = load_and_merge_data(
            toxicity_file, labels_file, smiles_col, toxicity_col, chain_type_col
        )

        # 2. 准备绘图数据
        straight_data, branched_data, straight_mean, branched_mean, \
            straight_median, branched_median, t_stat, p_value = prepare_plot_data(
            merged_df, toxicity_col, chain_type_col
        )

        # 3. 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 4. 生成所有图形
        print("\n" + "=" * 60)
        print("开始生成图形...")
        print("=" * 60)

        print("1. 创建箱线图...")
        create_box_plot(straight_data, branched_data, p_value, output_dir)

        print("2. 创建小提琴图...")
        create_violin_plot(straight_data, branched_data, straight_mean, branched_mean, output_dir)

        print("3. 创建直方图...")
        create_histogram(straight_data, branched_data, output_dir)

        print("4. 创建CDF图...")
        create_cdf_plot(straight_data, branched_data, straight_median, branched_median, output_dir)

        # 5. 保存合并后的数据（可选）
        merged_df.to_csv(f"{output_dir}/merged_data.csv", index=False)
        print(f"  合并后的数据已保存到: {output_dir}/merged_data.csv")

        print(f"\n所有图形已保存到 '{output_dir}' 目录!")
        print("生成的文件:")
        print(f"  - {output_dir}/box_plot.png")
        print(f"  - {output_dir}/violin_plot.png")
        print(f"  - {output_dir}/histogram.png")
        print(f"  - {output_dir}/cdf_plot.png")
        print(f"  - {output_dir}/merged_data.csv")

        # 6. 生成数据摘要报告
        with open(f"{output_dir}/data_summary.txt", "w") as f:
            f.write("毒性数据摘要报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"数据来源:\n")
            f.write(f"  毒性文件: {toxicity_file}\n")
            f.write(f"  标签文件: {labels_file}\n\n")
            f.write(f"样本统计:\n")
            f.write(f"  直链样本数: {len(straight_data)}\n")
            f.write(f"  支链样本数: {len(branched_data)}\n")
            f.write(f"  总样本数: {len(straight_data) + len(branched_data)}\n\n")
            f.write(f"描述性统计:\n")
            f.write(f"  直链均值: {straight_mean:.4f}\n")
            f.write(f"  直链标准差: {np.std(straight_data):.4f}\n")
            f.write(f"  直链中位数: {straight_median:.4f}\n")
            f.write(f"  支链均值: {branched_mean:.4f}\n")
            f.write(f"  支链标准差: {np.std(branched_data):.4f}\n")
            f.write(f"  支链中位数: {branched_median:.4f}\n\n")
            f.write(f"统计检验:\n")
            if p_value is not None:
                f.write(f"  t检验结果: t = {t_stat:.4f}, p = {p_value:.6f}\n")
                if p_value < 0.05:
                    f.write(f"  结论: 两组数据有显著差异 (p < 0.05)\n")
                else:
                    f.write(f"  结论: 两组数据无显著差异 (p ≥ 0.05)\n")
            f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"  数据摘要报告: {output_dir}/data_summary.txt")

    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
        print("请检查文件路径是否正确。")
    except ValueError as e:
        print(f"错误: {e}")
        print("请检查CSV文件中的列名是否正确。")
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback

        traceback.print_exc()