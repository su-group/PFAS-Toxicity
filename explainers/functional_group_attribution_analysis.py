#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能基团归因分析与文本自动避让散点图
参考：Atr_fig6_260530.py
修复：确保归因数据与包含基团的分子严格对应
新增：将小提琴图和散点图合并到同一个figure中展示
"""

import os
import sys
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
import pandas as pd
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 禁用输出缓冲，确保实时显示
print("=== 功能基团归因分析脚本启动 ===")
sys.stdout.flush()

# --- 配置 ---
DATA_PATH = r'F:\GNN-pro\data_process_class\step3_OECD_Class_Enhanced_v2_with_CF.csv'
PREDICTIONS_CACHE_PATH = r'F:\GNN-pro\data_process_class\predict_oecd_cf2_GIN510_v3.csv'
PRETRAINED_MODEL_PATH = r"F:\GNN-pro\scripts\outputs\GIN_20260107-094723\GIN_TL_Start_cf2_350_task_20260528-184435\transfer_learned_best_model.pth"

MODEL_CONFIG = {
    "model_type": "GIN",
    "params": {
        "hidden_dim": 256, "num_layers": 8, "dropout": 0.2,
        "prediction_tasks": 1, "input_dim": 120,
    },
}

NUM_EPOCHS_FOR_EXPLANATION = 50

# --- 修复的图构建器 ---
import periodictable

class FixedMoleculeGraphBuilder:
    def __init__(self, max_atoms=512, input_dim=120):
        self.max_atoms = max_atoms
        self.input_dim = input_dim
        self.atom_types = [el.symbol for el in periodictable.elements]
        self.atom_type_to_idx = {atom: idx for idx, atom in enumerate(self.atom_types)}

    def smiles_to_pyg_data(self, smiles: str) -> Data:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"无法解析SMILES: {smiles}")

            mol = Chem.AddHs(mol)
            num_atoms = mol.GetNumAtoms()

            atom_features = torch.zeros((self.max_atoms, self.input_dim))

            for i in range(min(num_atoms, mol.GetNumAtoms())):
                atom = mol.GetAtomWithIdx(i)
                symbol = atom.GetSymbol()
                if symbol in self.atom_type_to_idx:
                    idx = self.atom_type_to_idx[symbol]
                    if idx < self.input_dim - 1:
                        atom_features[i, idx] = 1
                    else:
                        atom_features[i, -1] = 1
                else:
                    atom_features[i, -1] = 1

            for i in range(num_atoms, self.max_atoms):
                atom_features[i, -1] = 1

            adjacency_matrix = torch.zeros((self.max_atoms, self.max_atoms))
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if i < self.max_atoms and j < self.max_atoms:
                    bond_type = bond.GetBondTypeAsDouble()
                    adjacency_matrix[i, j] = bond_type
                    adjacency_matrix[j, i] = bond_type

            edge_index = adjacency_matrix.nonzero().t().contiguous()
            edge_attr = adjacency_matrix[edge_index[0], edge_index[1]]

            data = Data(x=atom_features.float(), edge_index=edge_index.long(), edge_attr=edge_attr.float())
            return data
        except Exception as e:
            raise ValueError(f"处理SMILES {smiles} 时出错: {str(e)}")

# --- 功能基团 SMARTS 模式 ---
# 注意：使用更精确的模式来区分不同的氟化基团
FUNCTIONAL_GROUP_SMARTS = {
    "-CF3": ("[#6](F)(F)F", "Halogen"),
    "-CF2-": ("[#6;!$(C(F)(F)F)](F)(F)", "Halogen"),  # 排除 CF3，只匹配 CF2
    "-CXF-": ("[#6;!$(C(-[#9])(-[#9]))](-[#9])", "Halogen"),
    "-OH": ("[OH]", "Polar"),
    "-COOH": ("C(=O)[OH]", "Polar"),
    "-C=O": ("C=O", "Polar"),
    # "-N+": ("[N+]", "Polar"),
    "-NH2": ("[N;H2]", "Polar"),
    "-SH": ("[SH]", "Polar"),
    "-SO3H": ("S(=O)(=O)[OH]", "Polar"),
    "-PO3H2": ("P(=O)([OH])[OH]", "Polar"),
    "-Cl": ("[Cl]", "Halogen"),
    "-Br": ("[Br]", "Halogen"),
    "-I": ("[I]", "Halogen"),
    "-NO2": ("N(=O)=O", "Polar"),
    "-CN": ("C#N", "Polar"),
    "C=C": ("C=C", "Hydrophobic"),
    "C≡C": ("C#C", "Hydrophobic"),
    "-Ar": ("a", "Hydrophobic"),
    "-O-": ("C-O-C", "Polar"),
    "-COO-": ("C(=O)-O-C", "Polar"),
    "-CON-": ("C(=O)-N", "Polar"),
    "-SO2-": ("S(=O)(=O)", "Polar"),
    "-PO4-": ("P(=O)([O-])[O-]", "Polar"),
    "C=N": ("C=N", "Polar"),
    "-SO-": ("S(=O)", "Polar"),
    "-S-": ("C-S-C", "Hydrophobic"),
}

def get_functional_groups_for_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return {}
    fg_dict = {}
    for fg_name, (smarts_pattern, category) in FUNCTIONAL_GROUP_SMARTS.items():
        substructure = Chem.MolFromSmarts(smarts_pattern)
        if substructure is None: continue
        matches = mol.GetSubstructMatches(substructure)
        if matches:
            atom_indices = set()
            for match in matches:
                for idx in match: atom_indices.add(idx)
            if atom_indices: fg_dict[fg_name] = list(atom_indices)
    return fg_dict

# --- 加载预训练模型 ---
def load_pretrained_model(model_path, new_prediction_tasks):
    print(f"正在加载迁移学习后的模型: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    loaded_model_state = checkpoint.get('model_state_dict', checkpoint)
    
    from models.gin_model import MoleculeGIN
    model = MoleculeGIN(
        hidden_dim=MODEL_CONFIG["params"]["hidden_dim"],
        num_layers=MODEL_CONFIG["params"]["num_layers"],
        dropout=MODEL_CONFIG["params"]["dropout"],
        prediction_tasks=new_prediction_tasks,
        input_dim=MODEL_CONFIG["params"]["input_dim"],
    )
    model.load_state_dict(loaded_model_state, strict=False)
    print("模型权重加载成功")
    return model

# --- 模型包装器 ---
class ModelWrapper(nn.Module):
    def __init__(self, original_model):
        super(ModelWrapper, self).__init__()
        self.model = original_model
    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        return output[0] if isinstance(output, tuple) else output

# --- 链长匹配毒性差异计算 ---
def calculate_matched_delta_toxicity(mol_data_list, fg_name):
    df_temp = pd.DataFrame(mol_data_list)
    with_fg = df_temp[df_temp['has_' + fg_name] == True]
    without_fg = df_temp[df_temp['has_' + fg_name] == False]
    if with_fg.empty or without_fg.empty: return 0.0
    matched_pairs = []
    all_lengths = set(with_fg['total_CF'].unique()).union(set(without_fg['total_CF'].unique()))
    for cf_len in all_lengths:
        tox_with = with_fg[with_fg['total_CF'] == cf_len]['toxicity']
        tox_without = without_fg[without_fg['total_CF'] == cf_len]['toxicity']
        if not tox_with.empty and not tox_without.empty:
            matched_pairs.append(tox_with.mean() - tox_without.mean())
    return np.mean(matched_pairs) if matched_pairs else 0.0

# --- 文本避让算法 ---
def place_labels_with_avoidance(ax, x_coords, y_coords, labels, label_width=70, label_height=24):
    """
    智能放置标签，避免重叠（包括标签之间和标签与数据点之间）
    :param ax: 坐标轴对象
    :param x_coords: x坐标数组
    :param y_coords: y坐标数组
    :param labels: 标签数组
    :param label_width: 标签宽度（像素）
    :param label_height: 标签高度（像素）
    """
    # 获取所有数据点的像素坐标（用于检测与数据点的重叠）
    all_points_pixels = []
    for x, y in zip(x_coords, y_coords):
        display_coords = ax.transData.transform([x, y])
        all_points_pixels.append((display_coords[0], display_coords[1]))
    
    # 存储已放置的标签位置
    placed_boxes = []
    
    # 按距离原点的距离排序，远的先放（重要的标签先放）
    points_with_dist = [(x, y, label, (x**2 + y**2)**0.5) 
                        for x, y, label in zip(x_coords, y_coords, labels)]
    points_with_dist.sort(key=lambda p: p[3], reverse=True)
    
    for x, y, label, _ in points_with_dist:
        # 转换为像素坐标
        display_coords = ax.transData.transform([x, y])
        px, py = display_coords[0], display_coords[1]
        
        # 扩展偏移方向，提供更多候选位置
        directions = [
            (30, 30), (-30, 30), (30, -30), (-30, -30),
            (45, 15), (-45, 15), (45, -15), (-45, -15),
            (15, 45), (-15, 45), (15, -45), (-15, -45),
            (50, 5), (-50, 5), (50, -5), (-50, -5),
            (5, 50), (-5, 50), (5, -50), (-5, -50),
            (25, 0), (-25, 0), (0, 25), (0, -25),
            (40, 25), (-40, 25), (40, -25), (-40, -25),
            (25, 40), (-25, 40), (25, -40), (-25, -40)
        ]
        
        best_xytext = None
        min_overlap = float('inf')
        
        for dx, dy in directions:
            # 计算候选位置
            candidate_x = px + dx
            candidate_y = py + dy
            candidate_box = (candidate_x - label_width/2, candidate_y - label_height/2,
                           candidate_x + label_width/2, candidate_y + label_height/2)
            
            # 检查与已放置标签的重叠
            overlap = 0
            for placed_box in placed_boxes:
                dx_overlap = min(candidate_box[2], placed_box[2]) - max(candidate_box[0], placed_box[0])
                dy_overlap = min(candidate_box[3], placed_box[3]) - max(candidate_box[1], placed_box[1])
                overlap += max(0, dx_overlap) * max(0, dy_overlap)
            
            # 检查与数据点的重叠（数据点视为半径12像素的圆）
            point_radius = 12
            for (point_x, point_y) in all_points_pixels:
                # 计算点到矩形中心的距离
                dist_sq = (point_x - candidate_x)**2 + (point_y - candidate_y)**2
                # 如果点在标签区域内，增加重叠惩罚
                if dist_sq < (label_width/2 + point_radius)**2:
                    overlap += 100  # 惩罚值
            
            if overlap < min_overlap:
                min_overlap = overlap
                best_xytext = (dx, dy)
                if overlap == 0:
                    break  # 找到完全不重叠的位置，停止搜索
        
        if best_xytext:
            dx, dy = best_xytext
            ax.annotate(
                label,
                (x, y),
                xytext=(dx, dy),
                textcoords='offset points',
                fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='gray', linewidth=0.5),
                arrowprops=dict(
                    arrowstyle='->',
                    connectionstyle='arc3,rad=0.15',
                    alpha=0.5,
                    color='gray',
                    linewidth=0.6
                ),
                zorder=10
            )
            # 记录已放置的标签位置
            placed_boxes.append((px + dx - label_width/2, py + dy - label_height/2,
                               px + dx + label_width/2, py + dy + label_height/2))

# --- 合并绘图函数：同时生成小提琴图和散点图 ---
def plot_combined_figures(fg_data, mol_data_list, safe_name, output_dir):
    """
    在同一个figure中绘制小提琴图和散点图
    左图：小提琴图（归因分布）
    右图：散点图（归因-毒性关系）
    """
    try:
        # 准备数据
        plot_data = [{'Functional Group': fg, 'Attribution Value': d['attr']} 
                    for fg, data_list in fg_data.items() 
                    for d in data_list]
        
        if not plot_data:
            print("  [WARNING] 没有可用的归因数据进行绘图")
            return
        
        plot_df = pd.DataFrame(plot_data)
        mean_attributions = plot_df.groupby('Functional Group')['Attribution Value'].mean().reset_index()
        mean_dict = dict(zip(mean_attributions['Functional Group'], mean_attributions['Attribution Value']))
        sorted_fg = mean_attributions.sort_values('Attribution Value', ascending=False)['Functional Group'].tolist()
        plot_df['Functional Group'] = pd.Categorical(plot_df['Functional Group'], categories=sorted_fg, ordered=True)
        
        # 计算散点图数据
        impact_rows = []
        for fg in FUNCTIONAL_GROUP_SMARTS:
            if len(fg_data[fg]) > 0:
                tox_vals_with = [mol_data_list[i]['toxicity'] for i, m in enumerate(mol_data_list) if m[f'has_{fg}']]
                tox_vals_without = [mol_data_list[i]['toxicity'] for i, m in enumerate(mol_data_list) if not m[f'has_{fg}']]
                if tox_vals_with and tox_vals_without:
                    mean_attr = np.mean([d['attr'] for d in fg_data[fg]])
                    delta_tox = np.mean(tox_vals_with) - np.mean(tox_vals_without)
                    impact_rows.append({
                        'Group': fg,
                        'Mean_Attribution': mean_attr,
                        'Δ_Toxicity': delta_tox,
                        'N_with': len(tox_vals_with)
                    })
        
        # 创建1x2的subplot布局
        fig, axes = plt.subplots(1, 2, figsize=(24, max(8, len(sorted_fg) * 0.5)))
        
        # 左图：小提琴图
        ax1 = axes[0]
        sns.violinplot(data=plot_df, x='Attribution Value', y='Functional Group', 
                      inner='quartile', palette="Set2", order=sorted_fg, 
                      scale="width", cut=0, ax=ax1)
        ax1.set_xlabel("Normalized Attribution Value", fontsize=14)
        ax1.set_ylabel("Molecular Fragment", fontsize=14)
        ax1.axvline(x=0, color='red', linestyle='--', label='Zero Attribution')
        ax1.legend(loc='upper left')
        for i, fg in enumerate(sorted_fg):
            if fg in mean_dict:
                ax1.text(mean_dict[fg], i, f"{mean_dict[fg]:.4f}", va='center', ha='center', 
                        fontsize=8, fontweight='bold', color='white',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7), zorder=5)
        ax1.set_title('(A) Attribution Distribution', fontsize=16, fontweight='bold', pad=10)
        
        # 右图：散点图（带文本避让）
        ax2 = axes[1]
        if impact_rows:
            impact_df = pd.DataFrame(impact_rows)
            impact_df = impact_df[impact_df['N_with'] >= 5]  # 筛选有意义的基团
            
            if len(impact_df) > 0:
                # 绘制散点
                scatter = ax2.scatter(
                    impact_df['Mean_Attribution'],
                    impact_df['Δ_Toxicity'],
                    s=impact_df['N_with'] * 5,
                    c=impact_df['Δ_Toxicity'],
                    cmap='RdBu_r',
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=1
                )
                
                # 绘制参考线
                ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
                ax2.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
                
                # 设置坐标轴
                ax2.set_xlabel('Mean Normalized Attribution', fontsize=14)
                ax2.set_ylabel('Δ Toxicity (with - without)', fontsize=14)
                
                # 添加颜色条
                cbar = plt.colorbar(scatter, ax=ax2)
                cbar.set_label('Δ Toxicity', fontsize=12)
                
                # 使用智能文本避让算法添加标签
                x_coords = impact_df['Mean_Attribution'].values
                y_coords = impact_df['Δ_Toxicity'].values
                labels = impact_df['Group'].values
                
                place_labels_with_avoidance(ax2, x_coords, y_coords, labels)
                
                ax2.set_title('(B) Attribution vs Toxicity Impact', fontsize=16, fontweight='bold', pad=10)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存组合图
        combined_path = os.path.join(output_dir, f"{safe_name}_combined_analysis.png")
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] 组合分析图已保存: {combined_path}")
        
        # 同时保存单独的小提琴图
        violin_path = os.path.join(output_dir, f"{safe_name}_global_attr_violin.png")
        fig_violin, ax_v = plt.subplots(figsize=(12, max(8, len(sorted_fg) * 0.5)))
        sns.violinplot(data=plot_df, x='Attribution Value', y='Functional Group', 
                     inner='quartile', palette="Set2", order=sorted_fg, 
                     scale="width", cut=0, ax=ax_v)
        ax_v.set_xlabel("Normalized Attribution Value", fontsize=16)
        ax_v.set_ylabel("Molecular Fragment", fontsize=16)
        ax_v.axvline(x=0, color='red', linestyle='--', label='Zero Attribution')
        ax_v.legend(loc='upper left')
        for i, fg in enumerate(sorted_fg):
            if fg in mean_dict:
                ax_v.text(mean_dict[fg], i, f"{mean_dict[fg]:.4f}", va='center', ha='center',
                         fontsize=10, fontweight='bold', color='white',
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7), zorder=5)
        plt.tight_layout()
        plt.savefig(violin_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] 小提琴图已保存: {violin_path}")
        
        # 同时保存单独的散点图
        scatter_path = os.path.join(output_dir, f"{safe_name}_attr_tox_scatter.png")
        fig_scatter, ax_s = plt.subplots(figsize=(12, 10))
        if impact_rows:
            impact_df = pd.DataFrame(impact_rows)
            impact_df = impact_df[impact_df['N_with'] >= 5]
            
            if len(impact_df) > 0:
                scatter = ax_s.scatter(
                    impact_df['Mean_Attribution'],
                    impact_df['Δ_Toxicity'],
                    s=impact_df['N_with'] * 5,
                    c=impact_df['Δ_Toxicity'],
                    cmap='RdBu_r',
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=1
                )
                
                ax_s.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
                ax_s.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
                ax_s.set_xlabel('Mean Normalized Attribution', fontsize=14)
                ax_s.set_ylabel('Δ Toxicity (with - without)', fontsize=14)
                ax_s.set_title('Functional Group Attribution vs Toxicity Impact', fontsize=16, fontweight='bold', pad=20)
                
                cbar = plt.colorbar(scatter, ax=ax_s)
                cbar.set_label('Δ Toxicity', fontsize=12)
                
                x_coords = impact_df['Mean_Attribution'].values
                y_coords = impact_df['Δ_Toxicity'].values
                labels = impact_df['Group'].values
                
                place_labels_with_avoidance(ax_s, x_coords, y_coords, labels)
        
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] 散点图已保存: {scatter_path}")
        
    except Exception as e:
        print(f"  [ERROR] 组合图生成失败: {e}")
        import traceback
        traceback.print_exc()

# --- 核心分析函数 ---
def analyze_functional_groups_by_config(df, model, device, experiment_configs, output_dir="./functional_group_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    
    for config in experiment_configs:
        name = config["name"]
        condition = config["condition"]
        safe_name = name.replace(':', '_').replace('/', '_').replace('\\', '_')
        
        print(f"\n{'='*80}")
        print(f"开始分析实验: {name}")
        print(f"{'='*80}")
        filtered_df = df.query(condition)
        if filtered_df.empty: 
            print(f"  [WARNING] 没有符合条件的分子，跳过")
            continue
        print(f"  [OK] 符合条件的分子数量: {len(filtered_df)}")

        # 存储：对于每个基团，存储 (归因值, 毒性, 碳氟长度) 的三元组
        fg_data = {fg: [] for fg in FUNCTIONAL_GROUP_SMARTS}
        # 存储所有分子的基本数据
        mol_data_list = []

        graph_builder = FixedMoleculeGraphBuilder(max_atoms=512, input_dim=MODEL_CONFIG["params"]["input_dim"])

        print(f"\n  步骤 1/3: 开始运行 GNNExplainer 分析 ({len(filtered_df)} 个分子)...")
        success_count = 0
        fail_count = 0
        
        for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), 
                          desc="  分析分子", unit="mol", ncols=80):
            mol_smiles = row['SMILES']
            mol_tox = row.get('predicted_toxicity', np.nan)
            mol_cf = row.get('total_CF', np.nan)
            
            if pd.isna(mol_tox) or pd.isna(mol_cf): 
                fail_count += 1
                continue

            try:
                example_data = graph_builder.smiles_to_pyg_data(mol_smiles)
                if example_data is None: 
                    fail_count += 1
                    continue
                example_data = example_data.to(device)
                if example_data.batch is None: 
                    example_data.batch = torch.zeros(example_data.num_nodes, dtype=torch.long, device=device)

                model_for_explainer = ModelWrapper(model).to(device)
                explainer = Explainer(
                    model=model_for_explainer,
                    algorithm=GNNExplainer(epochs=NUM_EPOCHS_FOR_EXPLANATION),
                    explanation_type="model",
                    node_mask_type="attributes",
                    edge_mask_type=None,
                    model_config=dict(mode="regression", task_level="graph", return_type="raw")
                )
                explanation = explainer(example_data.x.to(device), example_data.edge_index.to(device), batch=example_data.batch.to(device))
                node_mask_raw = explanation.node_mask
                node_mask_cpu = node_mask_raw.mean(dim=1).cpu().detach().numpy() if node_mask_raw.dim() > 1 else node_mask_raw.cpu().detach().numpy()

                fg_dict = get_functional_groups_for_molecule(mol_smiles)
                total_attr = np.sum(np.abs(node_mask_cpu))
                
                # 收集当前分子的所有基团归因
                for fg_name, atom_indices in fg_dict.items():
                    valid_indices = [i for i in atom_indices if i < len(node_mask_cpu)]
                    if valid_indices:
                        mean_attr = np.mean([node_mask_cpu[i] for i in valid_indices])
                        norm_attr = (mean_attr / total_attr) if total_attr > 0 else 0.0
                        # 关键修复：只对包含该基团的分子存储归因值！
                        fg_data[fg_name].append({
                            'attr': norm_attr,
                            'tox': mol_tox,
                            'cf': mol_cf
                        })

                # 收集所有分子的基本数据
                mol_info = {'total_CF': mol_cf, 'toxicity': mol_tox}
                # 先初始化所有基团为False
                for fg_name in FUNCTIONAL_GROUP_SMARTS:
                    mol_info[f'has_{fg_name}'] = False
                # 然后设置实际存在的基团为True
                for fg_name in fg_dict.keys():
                    if fg_name in FUNCTIONAL_GROUP_SMARTS:
                        mol_info[f'has_{fg_name}'] = True
                mol_data_list.append(mol_info)
                success_count += 1
                
            except Exception as e:
                fail_count += 1
                continue

        print(f"  [OK] GNNExplainer 分析完成: 成功 {success_count}, 失败 {fail_count}")

        # 1. 计算链长匹配后的毒性差异
        print(f"\n  步骤 2/3: 正在计算匹配链长后的毒性差异...")
        matched_impact = {}
        for fg_name in tqdm(FUNCTIONAL_GROUP_SMARTS, desc="  计算毒性差异", unit="fg", ncols=80):
            matched_impact[fg_name] = calculate_matched_delta_toxicity(mol_data_list, fg_name)
        
        # 打印关键基团对比
        print("\n  关键基团统计:")
        for fg in ["-CF2-", "-CF3", "-CXF-", "-OH", "-Br"]:
            orig_mean = np.mean([d['attr'] for d in fg_data[fg]]) if fg_data[fg] else 0
            print(f"    -> {fg:6s} | 样本数: {len(fg_data[fg]):4d} | 原始归因均值: {orig_mean:.4f} | Matched ΔTox: {matched_impact.get(fg, 0):+.4f}")

        # 2. 同时生成小提琴图和散点图
        print(f"\n  步骤 3/3: 生成归因分析图（小提琴图 + 散点图）...")
        plot_combined_figures(fg_data, mol_data_list, safe_name, output_dir)

# --- 实验配置生成器 ---
def AutoToxLengthScanBySpecificCombinations(df, target_combinations, base_conditions=None):
    experiments = []
    working_df = df.query(base_conditions) if base_conditions else df
    conditions_list = []
    for fc, sc in target_combinations:
        fc_escaped = fc.replace("'", "\\'")
        sc_escaped = sc.replace("'", "\\'")
        conditions_list.append(f"(first_class == '{fc_escaped}' and second_class == '{sc_escaped}')")
    combined_condition = " or ".join(conditions_list)
    full_query = f"({combined_condition})"
    if base_conditions:
        full_query = f"({base_conditions}) and ({combined_condition})"
    subset = working_df.query(full_query)
    num_molecules = len(subset)
    print(f"为特定组合分析准备数据: {num_molecules} 个分子匹配条件")
    if num_molecules > 0:
        name_parts = [f"{fc.replace(' ', '_')}_vs_{sc.replace(' ', '_')}" for fc, sc in target_combinations[:3]]
        name_suffix = "_etc" if len(target_combinations) > 3 else ""
        experiment_name = f"{'_and_'.join(name_parts)}{name_suffix}_Toxicity_vs_CF_Length"
        experiment_config = {
            "name": experiment_name,
            "condition": full_query,
            "analysis_type": "scatter_plot",
            "x_var": "total_CF",
            "y_var": "predicted_toxicity",
            "hue_var": "second_class",
            "style_var": "functional_group",
            "R2": False,
        }
        experiments.append(experiment_config)
        print(f"成功为 {len(target_combinations)} 个特定类别组合生成了实验配置")
    return experiments

# --- 主程序 ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model = load_pretrained_model(PRETRAINED_MODEL_PATH, new_prediction_tasks=1).to(device)
    print("模型加载完成")

    print("正在加载结构化数据...")
    df_enhanced = pd.read_csv(DATA_PATH)
    df_enhanced['total_CF'] = df_enhanced['CF2'] + df_enhanced['CF3'] + df_enhanced['CFX']
    print(f"数据加载完成，共 {len(df_enhanced)} 行")

    predictions_df = pd.DataFrame()
    if os.path.exists(PREDICTIONS_CACHE_PATH):
        try:
            predictions_df = pd.read_csv(PREDICTIONS_CACHE_PATH)
            print(f"缓存加载成功，共 {len(predictions_df)} 条记录")
        except:
            print("缓存加载失败")
    df_enhanced_with_pred = df_enhanced.merge(predictions_df[['SMILES', 'predicted_toxicity']], on='SMILES', how='left')

    print("\n--- 开始生成实验配置并进行功能基团归因分析 ---")
    TARGET_COMBINATIONS = [
        ("PFAA precursors", "SFAenes"), ("PFAA precursors", "PFAenes"), ("PFAA precursors", "HFEs"),
        ("PFAAs", "PFECAs"), ("PFAA precursors", "PFAIs"),
        ("PFAA precursors", "HFCs"),
        ("Polyfluoroalkyl acids", "PolyFCAs"),
        ("PFAAs", "PFCA-ester derivatives"),
        ("Other PFASs", "Polyfluoroalkanes"), ("Other PFASs", "PFPEs"),
        ("PFAA precursors", "PolyFEACs"), ("PFAA precursors", "n:2 fluorotelomer-based substances"),
        ("PFAA precursors", "PASF-based substances"), ("PFAA precursors", "PFAK derivatives"),
        ("PFAA precursors", "PolyFACs")
    ]
    experiment_configs = AutoToxLengthScanBySpecificCombinations(
        df_enhanced_with_pred,
        target_combinations=TARGET_COMBINATIONS,
        base_conditions=None
    )
    analyze_functional_groups_by_config(df_enhanced_with_pred, model, device, experiment_configs)
    print("\n所有分析执行完毕！")
