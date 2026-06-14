#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN-Based Structure-Activity Relationship (SAR) Analysis with GNNexplainer
Using a Pre-trained and Transferred GNN Model for Toxicity Prediction
WITH PREDICTION CACHING
"""
import os
import sys
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.explain import GNNExplainer
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import mplcursors # 用于交互式光标
import seaborn as sns
import textwrap # 用于文本换行
from scipy.stats import pearsonr

# --- 1. 添加项目根目录到系统路径 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# --- 2. 导入项目模块 ---
from config import get_model_save_path
from dataset.molecule_dataset import MoleculeDataset
from models.gnn_model import MoleculeGNN, UncertaintyLoss
from models.gat_model import MoleculeGAT
from models.gin_model import MoleculeGIN
from models.jknet_model import MoleculeJKNet
from dataset.graph_builder import MoleculeGraphBuilder
from torch_geometric.explain import Explainer, GNNExplainer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# --- 2. 配置 ---
# --- 2.1. 路径配置 ---
DATA_PATH = r'F:\GNN-pro\data_process_class\step3_OECD_Class_Enhanced_v2_with_CF.csv'
PREDICTIONS_CACHE_PATH = r'F:\GNN-pro\data_process_class\predict_oecd_cf2_GIN510_v3.csv'
PRETRAINED_MODEL_PATH = r"F:\GNN-pro\scripts\outputs\GIN_20260107-094723\GIN_TL_Start_cf2_510_task_20260528-184435\transfer_learned_best_model.pth"
# --- 2.2. 模型配置 (与迁移学习训练时保持一致) ---
MODEL_CONFIG = {
    "model_type": "GIN", # 假设是 GNN
    "supported_types": ["GNN", "GAT", "GIN", "Transformer", "JKNet"],
    "params": {
        "hidden_dim": 256,
        "num_layers": 8,
        "dropout": 0.2,
        "prediction_tasks": 1,
        "input_dim": 120, # 重要：与训练时一致
    },
    "type_specific_params": {
        "GNN": {"model_type": "GCN"},
        "GAT": {"num_heads": 4},
        "JKNet": {"jk_mode": "cat"},
        "Transformer": {"num_heads": 8}
    }
}
# --- 2.3. 可变参数---
TARGET_MEAN = 2.9348 #模型反标准化参数，换模型记得换
TARGET_STD = 1.0857   #模型反标准化参数，换模型记得换
num_epochs = 200#解释器参数，记得更改，小快大细
# --- 2.4.设置中文字体 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用 SimHei 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# --- 3. 加载数据 ---
print("正在加载结构化数据...")
df_enhanced = pd.read_csv(DATA_PATH)
# 创建含F碳链长列
df_enhanced['total_CF'] = df_enhanced['CF2'] + df_enhanced['CF3'] + df_enhanced['CFX']
# 添加一个辅助列用于缓存
df_enhanced['cache_key'] = df_enhanced['SMILES'] # 使用 SMILES 作为唯一标识
print(f"数据加载完成，共 {len(df_enhanced)} 行。")
# --- 4. 定义加载预训练模型的函数 (根据您的迁移学习代码调整) ---
def load_pretrained_model(model_path, new_prediction_tasks):
    """
    加载迁移学习后的模型。假设模型结构和输出层已调整好。
    """
    print(f"正在加载迁移学习后的模型: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    loaded_model_state = checkpoint.get('model_state_dict', checkpoint)
    # 重建模型结构 (必须与迁移学习时的模型结构完全一致)
    original_model_type = MODEL_CONFIG["model_type"]
    original_prediction_tasks = MODEL_CONFIG["params"]["prediction_tasks"]
    input_dim = MODEL_CONFIG["params"]["input_dim"]
    hidden_dim = MODEL_CONFIG["params"]["hidden_dim"]
    num_layers = MODEL_CONFIG["params"]["num_layers"]
    dropout = MODEL_CONFIG["params"]["dropout"]
    type_specific_params = MODEL_CONFIG.get("type_specific_params", {})
    print(f"模型类型: {original_model_type}, 预测任务数: {new_prediction_tasks}")
    print(f"模型参数: input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}")
    # --- 根据模型类型重建模型 ---
    model_params = {
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'prediction_tasks': new_prediction_tasks,
        'input_dim': input_dim,
    }
    if original_model_type == "GNN":
        model_params['model_type'] = type_specific_params.get('GNN', {}).get('model_type', 'GCN')
        model = MoleculeGNN(**model_params)
    elif original_model_type == "GAT":
        model_params['num_heads'] = type_specific_params.get('GAT', {}).get('num_heads', 4)
        model = MoleculeGAT(**model_params)
    elif original_model_type == "GIN":
        model = MoleculeGIN(**model_params)
    elif original_model_type == "JKNet":
        model_params['jk_mode'] = type_specific_params.get('JKNet', {}).get('jk_mode', 'cat')
        model = MoleculeJKNet(**model_params)
    else:
        raise ValueError(f"不支持的原始模型类型: {original_model_type}")
    # 加载权重 (注意：如果模型结构与预训练模型不匹配，这里会失败)
    try:
        model.load_state_dict(loaded_model_state, strict=True)
        print("模型权重加载成功。")
    except Exception as e:
        print(f"模型权重加载失败: {e}")
        print("请确保模型结构与预训练模型完全一致。")
        sys.exit(1)
    model.prediction_tasks = new_prediction_tasks
    return model
# --- 5. 定义执行对比实验的函数 ---
# --- 5.1. 首先，定义模型包装器类 ---
class ModelWrapper(nn.Module):
    """
    A wrapper for the model that returns only the prediction part
    when the original model returns (prediction, uncertainty).
    """
    def __init__(self, original_model):
        super(ModelWrapper, self).__init__()
        self.model = original_model # Store the original model
    def forward(self, *args, **kwargs):
        """
        Forward pass that calls the original model and returns only the prediction.
        Assumes original model returns (prediction, uncertainty).
        """
        output = self.model(*args, **kwargs)
        if isinstance(output, tuple):
            prediction = output[0] # Take the first element (prediction)
        else:
            # If it's not a tuple, assume it's already the prediction
            prediction = output
        return prediction
# --- 5.2.  execute_dual_plot 函数 ---
def execute_dual_plot(exp_config, df):
    """
    Executes a dual plot for both absolute count and ratio.
    """
    condition = exp_config["condition"]
    filtered_df = df.query(condition) if condition else df
    # Calculate ratios
    if 'carbon_chain_length' not in filtered_df.columns:
        print(f"Warning: 'carbon_chain_length' column not found in data for dual plot. Cannot calculate ratios.")
        return
    total_carbon_col = 'carbon_chain_length'
    # Ensure total_carbon_col is not zero to avoid division by zero
    filtered_df = filtered_df[filtered_df[total_carbon_col] != 0].copy() # Work on a copy and filter out zeros
    # Calculate the specific ratio based on the x_var in the config
    ratio_var = exp_config["x_var"] # e.g., 'CFX_ratio' or 'Total_F_Ratio'
    # --- 修改：增加对 Total_F_Ratio 的特殊处理 ---
    if ratio_var == 'Total_F_Ratio':
        absolute_var = 'total_CF' # The absolute count for Total_F_Ratio is total_CF
    else:
        # For other ratios, use the original logic
        absolute_var = ratio_var.replace('_ratio', '') # e.g., 'CFX_ratio' -> 'CFX'
    valid_absolute_vars = ['CF2', 'CF3', 'CFX', 'total_CF']
    if absolute_var not in valid_absolute_vars:
        print(f"Warning: Cannot determine absolute variable from ratio '{ratio_var}'. Valid ratios should lead to one of: {valid_absolute_vars}")
        return
    # Calculate the specific ratio if it doesn't exist in the dataframe yet
    # This check is important because ratios might already exist from previous function calls
    if ratio_var not in filtered_df.columns:
        if absolute_var not in filtered_df.columns:
             print(f"Warning: Absolute variable '{absolute_var}' not found in data for dual plot. Cannot calculate ratio.")
             return
        filtered_df[ratio_var] = filtered_df[absolute_var] / filtered_df[total_carbon_col]
    # --- 获取 style 变量 ---
    style_var = exp_config.get("style_var", None)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    hue_var = exp_config.get("hue_var", "functional_group")
    # Plot 1: Absolute Count
    ax1 = axes[0]
    sns.scatterplot(
        data=filtered_df,
        x=absolute_var,
        y='predicted_toxicity',
        hue=hue_var,
        style=style_var, # Add style parameter
        palette="Set2",
        alpha=0.7,
        ax=ax1
    )
    ax1.set_title(f"{exp_config['title']} - Absolute Count ({absolute_var})")
    ax1.set_xlabel(absolute_var)
    ax1.set_ylabel('Predicted Toxicity')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # Add R² for absolute count plot if requested
    if exp_config.get("r", False):
        plot_data_valid_abs = filtered_df.dropna(subset=[absolute_var, 'predicted_toxicity'])
        if len(plot_data_valid_abs) >= 2:
            X_abs = plot_data_valid_abs[absolute_var].values.reshape(-1, 1)
            y_abs = plot_data_valid_abs['predicted_toxicity'].values
            reg_model_abs = LinearRegression()
            reg_model_abs.fit(X_abs, y_abs)
            x_fit_abs = np.linspace(X_abs.min(), X_abs.max(), 100).reshape(-1, 1)
            y_fit_abs = reg_model_abs.predict(x_fit_abs)
            r_abs, _ = pearsonr(plot_data_valid_abs[absolute_var], y_abs)
            ax1.plot(x_fit_abs, y_fit_abs, 'r-', linewidth=2, label=f'Linear Fit (r = {r_abs:.3f})')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # Plot 2: Ratio
    ax2 = axes[1]
    sns.scatterplot(
        data=filtered_df,
        x=ratio_var,
        y='predicted_toxicity',
        hue=hue_var,
        style=style_var, # Add style parameter
        palette="Set2",
        alpha=0.7,
        ax=ax2
    )
    ax2.set_title(f"{exp_config['title']} - Ratio ({ratio_var})")
    ax2.set_xlabel(ratio_var)
    ax2.set_ylabel('Predicted Toxicity')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # Add R² for ratio plot if requested
    if exp_config.get("r", False):
        plot_data_valid_rat = filtered_df.dropna(subset=[ratio_var, 'predicted_toxicity'])
        if len(plot_data_valid_rat) >= 2:
            X_rat = plot_data_valid_rat[ratio_var].values.reshape(-1, 1)
            y_rat = plot_data_valid_rat['predicted_toxicity'].values
            reg_model_rat = LinearRegression()
            reg_model_rat.fit(X_rat, y_rat)
            x_fit_rat = np.linspace(X_rat.min(), X_rat.max(), 100).reshape(-1, 1)
            y_fit_rat = reg_model_rat.predict(x_fit_rat)
            r_rat, _ = pearsonr(plot_data_valid_rat[ratio_var], y_rat)
            ax2.plot(x_fit_rat, y_fit_rat, 'r-', linewidth=2, label=f'Linear Fit (r = {r_rat:.3f})')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_name = exp_config["name"] + "_dual_plot.png"
    plt.savefig(plot_name, dpi=300)
    plt.close()
    print(f"双图保存为 {plot_name}")

# --- 5.3.  run_sar_experiment 函数 (修改版) ---
def run_sar_experiment(df, model, device, experiment_name, query_condition, exp_config=None, smiles_col='SMILES'):
    """
    执行一个 SAR 对比实验，包括毒性预测和 GNNexplainer 分析。
    Now also handles violin plot by CF bin and dual plot if analysis_type matches.
    """
    print(f"--- 开始实验: {experiment_name} ---")
    print(f"筛选条件: {query_condition}")
    # 1. 筛选分子
    if query_condition is not None: # Check if condition is provided
        try:
            subset = df.query(query_condition)
        except Exception as e:
            print(f"查询失败: {e}")
            print(f"尝试使用 pandas boolean indexing...")
            subset = pd.DataFrame() # Return empty DataFrame on error
    else:
        # If no condition is provided, use the full dataframe
        subset = df
        print("  未提供筛选条件，使用完整数据集。")
    if subset.empty:
        print("筛选结果为空，跳过此实验。")
        return
    print(f"筛选出 {len(subset)} 个分子。")
    # 2. 加载或计算预测毒性
    # 尝试从缓存文件加载
    predictions_df = pd.DataFrame()
    if os.path.exists(PREDICTIONS_CACHE_PATH):
        print(f"尝试从缓存文件加载预测结果: {PREDICTIONS_CACHE_PATH}")
        try:
            predictions_df = pd.read_csv(PREDICTIONS_CACHE_PATH)
            print(f"缓存加载成功，共 {len(predictions_df)} 条记录。")
        except Exception as e:
            print(f"缓存文件加载失败: {e}")
            predictions_df = pd.DataFrame()
    # 确定哪些 SMILES 需要预测
    cached_smiles = set(predictions_df['SMILES']) if not predictions_df.empty else set()
    all_smiles = set(subset[smiles_col])
    uncached_smiles = all_smiles - cached_smiles
    if uncached_smiles:
        print(f"发现 {len(uncached_smiles)} 个未缓存的分子，需要进行预测。")
        # 构建未缓存分子的图数据
        graph_builder = MoleculeGraphBuilder()
        uncached_subset = subset[subset[smiles_col].isin(uncached_smiles)]
        pyg_data_list = []
        uncached_smiles_list = []
        valid_original_indices = [] # 保存原始索引，用于后续与预测结果关联
        for relative_idx, (original_idx, row) in enumerate(uncached_subset.iterrows()): # 获取相对索引和原始索引
            smiles = row[smiles_col]
            try:
                pyg_data = graph_builder.smiles_to_pyg_data(smiles)
                if pyg_data is not None:
                    pyg_data_list.append(pyg_data)
                    uncached_smiles_list.append(smiles)
                    valid_original_indices.append(original_idx) # 保存原始索引
                else:
                    print(f"警告: 无法处理 SMILES {smiles} (来自 graph_builder)")
            except Exception as e:
                print(f"警告: 处理 SMILES {smiles} 时出错: {e}")
        if pyg_data_list:
            print(f"正在为 {len(pyg_data_list)} 个有效分子进行毒性预测...")
            data_loader = DataLoader(pyg_data_list, batch_size=len(pyg_data_list)) # 一次性加载所有
            model.eval() # 设置原始模型为评估模式
            with torch.no_grad(): # 禁用梯度计算以节省内存和加速
                for batch in data_loader:
                    batch = batch.to(device)
                    # 修正：模型调用，假设模型返回 (prediction, uncertainty)
                    toxicity_pred, _ = model(batch.x, batch.edge_index, batch.batch) # 只取预测值
                    # 反标准化
                    toxicity_pred_denorm = toxicity_pred * TARGET_STD + TARGET_MEAN
                    # 将预测结果转换为 numpy 数组
                    toxicity_values = toxicity_pred_denorm.cpu().numpy().flatten()
            # 创建新的预测结果 DataFrame
            # 使用保存的原始索引，确保与原始数据集的行对应
            new_predictions_df = pd.DataFrame({
                'SMILES': uncached_smiles_list,
                'predicted_toxicity': toxicity_values,
                'original_index': valid_original_indices # 添加原始索引列，用于后续验证或处理
            })
            # 合并到缓存 DataFrame
            predictions_df = pd.concat([predictions_df, new_predictions_df.drop(columns=['original_index'])], ignore_index=True) # 合并时去掉临时索引列
            # 保存更新后的缓存
            predictions_df.to_csv(PREDICTIONS_CACHE_PATH, index=False)
            print(f"更新后的预测结果已保存到缓存文件: {PREDICTIONS_CACHE_PATH}")
        else:
            print("没有有效的未缓存分子图用于预测。")
    # 3. 将缓存的预测结果合并到筛选数据子集
    print("正在将预测结果合并到数据子集...")
    # 使用 merge 保留 subset 的原始顺序和行数，未预测的分子毒性值为 NaN
    smiles_to_tox = dict(zip(predictions_df['SMILES'], predictions_df['predicted_toxicity']))
    subset['predicted_toxicity'] = subset['SMILES'].map(smiles_to_tox)
    # 检查是否有分子未成功预测（理论上不应该有，除非缓存逻辑有问题）
    missing_predictions = subset['predicted_toxicity'].isna().sum()
    if missing_predictions > 0:
        print(f"警告: 有 {missing_predictions} 个分子未能获取预测毒性值。")
    # --- Check for the specific analysis type and execute accordingly ---
    analysis_type = exp_config.get("analysis_type", "scatter_plot") # Default to scatter plot logic
    if analysis_type == "violin_plot_by_cf_bin":
        print(f"  检测到分析类型 '{analysis_type}'，执行 Violin Plot by CF Bin...")
        execute_global_plot_violin_by_cf_length(exp_config, subset) # Pass the filtered subset
        return # Exit after executing the violin plot
    elif analysis_type == "dual_plot": # New condition for dual plot
        print(f"  检测到分析类型 '{analysis_type}'，执行 Dual Plot (Absolute vs Ratio)...")
        execute_dual_plot(exp_config, subset) # Pass the data with predictions
        return # Exit after executing the dual plot
    # 5. 可视化结果 (毒性 vs 结构变量)
    print("正在生成毒性与结构变量的可视化结果...")
    # --- 初始化 r 变量 ---
    r = float('nan') # 初始化 r 为 NaN (Not a Number)
    if 'predicted_toxicity' in subset.columns:
        plot_data = subset.dropna(subset=['predicted_toxicity'])

        # --- 获取 style 变量 ---
        style_var = exp_config.get("style_var", None)

        plt.figure(figsize=(12, 8))
        # >>> 新增：动态获取 hue 变量 <<<
        hue_var = exp_config.get("hue_var", "functional_group") if exp_config else "functional_group"
        # 检查 hue_var 是否有效
        use_hue = (hue_var in plot_data.columns) and plot_data[hue_var].notna().any()
        # 检查 style_var 是否有效
        use_style = (style_var is not None) and (style_var in plot_data.columns) and plot_data[style_var].notna().any()

        if use_hue or use_style:
            # --- Prepare palette and legend handling ---
            # Determine unique categories for hue and style
            unique_hue_values = plot_data[hue_var].dropna().unique() if use_hue else []
            unique_style_values = plot_data[style_var].dropna().unique() if use_style else []

            # --- Determine Palette ---
            n_hue = len(unique_hue_values)
            if n_hue <= 20:
                palette = sns.color_palette("tab20", n_colors=n_hue)
            else:
                palette = sns.color_palette("husl", n_colors=n_hue)  # 更多类别时使用 husl

            # --- Create the plot ---
            scatter_plot = sns.scatterplot(
                data=plot_data,
                x=exp_config["x_var"], # Use the x_var from the config
                y='predicted_toxicity',
                hue=hue_var,
                style=style_var, # Add style parameter
                palette=palette if use_hue else None, # Use palette if hue is used
                alpha=0.7,
                legend='auto'
            )

            # --- Title and Labels ---
            title_hue_part = f" by {hue_var}" if use_hue else ""
            title_style_part = f" & {style_var}" if use_style else ""
            plt.title(f'{experiment_name}: Predicted Toxicity vs {exp_config["x_var"]}{title_hue_part}{title_style_part}', fontsize=14, fontweight='bold')
            plt.xlabel(f'{exp_config["x_var"]}', fontsize=12) # Use the x_var from the config
            plt.ylabel('Predicted Toxicity', fontsize=12)
            plt.grid(False)
            plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=8, frameon=True, shadow=True)
            # 交互式光标（保持原逻辑）
            def on_add(sel):
                index = sel.index
                row = plot_data.iloc[index]
                smi = row[smiles_col]
                smi_short = textwrap.fill(smi, width=40)
                # 动态获取 hue 值用于显示（可选）
                hue_val = row.get(hue_var, 'N/A') if use_hue and hue_var in row.index else 'N/A'
                # 动态获取 style 值用于显示（可选）
                style_val = row.get(style_var, 'N/A') if use_style and style_var in row.index else 'N/A'
                sel.annotation.set_text(
                    f"SMILES: {smi_short}\n"
                    f"Class: {row.get('first_class', 'N/A')} - {row.get('second_class', 'N/A')}\n"
                    f"Toxicity: {row['predicted_toxicity']:.4f}\n"
                    f"{exp_config['x_var']}: {row[exp_config['x_var']]:.4f}\n" # Use the x_var from the config
                    f"Total CF: {row['total_CF']:.0f}\n"
                    f"CF2: {row['CF2']}, CF3: {row['CF3']}, CFX: {row['CFX']}\n"
                    f"Chain: {row['chain_type']}\n"
                    f"Carbon Chain Length: {row['carbon_chain_length']}\n"
                    f"{hue_var if use_hue else 'No Hue'}: {hue_val}\n"
                    f"{style_var if use_style else 'No Style'}: {style_val}\n"
                    f"Ring Size: {row.get('ring_size', 'N/A')}\n"
                    f"num_id: {row['num_id']}\n"
                )
                sel.annotation.get_bbox_patch().set(alpha=0.8)
                sel.annotation.set_fontsize(8)
            cursor = mplcursors.cursor(scatter_plot, hover=True)
            cursor.connect("add", on_add)
        else:
            # 无有效 hue 或 style 变量，退化为单色图
            scatter_plot = plt.scatter(plot_data[exp_config["x_var"]], plot_data['predicted_toxicity'], alpha=0.7, color='blue') # Use the x_var from the config
            plt.title(f'{experiment_name}: Predicted Toxicity vs {exp_config["x_var"]}', fontsize=14, fontweight='bold')
            plt.xlabel(f'{exp_config["x_var"]}', fontsize=12) # Use the x_var from the config
            plt.ylabel('Predicted Toxicity', fontsize=12)
            plt.grid(False)
            # 单色图同样支持 R²
            def on_add(sel):
                index = sel.target.index
                row = plot_data.iloc[index]
                smi = row[smiles_col]
                smi_short = textwrap.fill(smi, width=40)
                sel.annotation.set_text(
                    f"SMILES: {smi_short}\n"
                    f"Class: {row.get('first_class', 'N/A')} - {row.get('second_class', 'N/A')}\n"
                    f"Toxicity: {row['predicted_toxicity']:.4f}\n"
                    f"{exp_config['x_var']}: {row[exp_config['x_var']]:.4f}\n" # Use the x_var from the config
                    f"Total CF: {row['total_CF']:.0f}\n"
                    f"CF2: {row['CF2']}, CF3: {row['CF3']}, CFX: {row['CFX']}\n"
                    f"Chain: {row['chain_type']}\n"
                    f"Carbon Chain Length: {row['carbon_chain_length']}\n"
                    f"Functional Group: {row.get('functional_group', 'N/A')}\n"
                    f"Ring Size: {row.get('ring_size', 'N/A')}\n"
                    f"num_id: {row['num_id']}\n"
                )
                sel.annotation.get_bbox_patch().set(alpha=0.8)
                sel.annotation.set_fontsize(8)
            cursor = mplcursors.cursor(scatter_plot, hover=True)
            cursor.connect("add", on_add)
        # >>> 统一添加 R² 拟合线（无论是否着色）<<<
        if exp_config is not None and exp_config.get("r", False):
            plot_data_valid = plot_data.dropna(subset=[exp_config["x_var"], 'predicted_toxicity'])
            if len(plot_data_valid) >= 2:
                X = plot_data_valid[exp_config["x_var"]].values.reshape(-1, 1)
                y = plot_data_valid['predicted_toxicity'].values

                # 线性拟合（用于画线）
                reg_model = LinearRegression()
                reg_model.fit(X, y)
                x_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                y_fit = reg_model.predict(x_fit)

                # 计算 Pearson r（用于展示关联强度）
                r, p_val = pearsonr(plot_data_valid[exp_config["x_var"]], y)

                plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Linear Fit (r = {r:.3f})')
                plt.legend()
        plt.show()
    else:
        print("没有有效的数据点用于绘制散点图。")
    # 6. 打印结果摘要
    valid_predictions = subset.dropna(subset=['predicted_toxicity'])
    if not valid_predictions.empty:
        print(f"--- 实验 {experiment_name} 结果摘要 ---")
        print(f"平均预测毒性: {valid_predictions['predicted_toxicity'].mean():.4f}")
        print(f"毒性标准差: {valid_predictions['predicted_toxicity'].std():.4f}")
        print(f"毒性范围: {valid_predictions['predicted_toxicity'].min():.4f} - {valid_predictions['predicted_toxicity'].max():.4f}")
        print(f"参与分析的有效分子数: {len(valid_predictions)}")
        print(f"Pearson r: {r:.3f}")
        print(f"------------------------")
    else:
        print(f"--- 实验 {experiment_name} 无有效预测结果 ---")

# --- 6. 实验参数配置 ---
def AutoToxLengthScanByCategory(df, base_conditions=None):
    """
    Generates experiment configurations dynamically based on the data.
    Only generates experiments for groups with more than 20 molecules.
    """
    experiments = []
    # Apply base conditions if provided
    working_df = df.query(base_conditions) if base_conditions else df
    # Example: Generate experiments for each first_class & second_class combo with straight chains
    # Filter for straight chains and non-null total_CF
    straight_df = working_df[working_df['chain_type'] == 'straight']
    straight_df = straight_df[straight_df['total_CF'].notna()]
    grouped = straight_df.groupby(['first_class', 'second_class'])
    for (fc, sc), group in grouped:
        # --- 新增：检查分子数量 ---
        num_molecules = len(group)
        if num_molecules <= 20:
            print(f"跳过组合 ({fc}, {sc})，因为只有 {num_molecules} 个分子 (<= 20)。")
            continue # 跳过这个组合，继续下一个
        # --- 原有条件：检查唯一 CF 长度数 ---
        unique_cf_lengths = group['total_CF'].nunique()
        if unique_cf_lengths >= 2: # Ensure enough data points for analysis
            experiment_config = {
                "name": f"{fc.replace(' ', '_')}_vs_{sc.replace(' ', '_')}_Toxicity_vs_CF_Length_Straight".replace(',', '').replace('&', 'and'),
                "condition": f"(first_class == '{fc}') & (second_class == '{sc}') & (chain_type == 'straight')",
                "analysis_type": "scatter_plot",
                "x_var": "total_CF",
                "y_var": "predicted_toxicity",
                "hue_var": "functional_group",
                "style_var": "chain_type", # Optional: e.g., shape by chain type
                "r": True,
                "title": f"{fc} ({sc}, Straight): Predicted Toxicity vs Total CF Groups"
            }
            experiments.append(experiment_config)
        else:
            print(f"跳过组合 ({fc}, {sc}) (分子数>{num_molecules})，因为只有 {unique_cf_lengths} 个唯一的 CF 长度值。")
    return experiments

def AutoToxLengthScanByFunctionGroup(df, base_conditions=None):
    """
    Generates experiment configurations dynamically based on Functional Group.
    Only generates experiments for Function Groups with more than 20 molecules.
    """
    experiments = []
    # Apply base conditions if provided
    working_df = df.query(base_conditions) if base_conditions else df
    # Filter for non-null total_CF and functional_group
    filtered_df = working_df[
        (working_df['total_CF'].notna()) &
        (working_df['functional_group'].notna()) &
        (working_df['functional_group'] != '')  # Exclude empty strings
        ]
    grouped = filtered_df.groupby(['functional_group'])
    for fg, group in grouped:
        # Check molecule count
        num_molecules = len(group)
        if num_molecules <= 20:
            print(f"Skipping Functional Group '{fg}', as it has only {num_molecules} molecules (<= 20).")
            continue  # Skip this group, process the next one
        # Check number of unique CF lengths within the group
        unique_cf_lengths = group['total_CF'].nunique()
        if unique_cf_lengths >= 2:  # Ensure enough data points for meaningful analysis
            # Determine chain types present in this functional group for the title
            chain_types_present = group['chain_type'].unique()
            chain_desc = ", ".join(sorted(chain_types_present))
            experiment_config = {
                "name": f"FuncGroup_{fg.replace(' ', '_').replace('/', '_or_')}_Toxicity_vs_CF_Length",
                "condition": f"(functional_group == '{fg}')",  # Base condition
                "analysis_type": "scatter_plot",
                "x_var": "total_CF",  # X-axis variable
                "y_var": "predicted_toxicity",  # Y-axis variable
                "hue_var": "second_class",  # Color by chain type within the functional group
                "style_var": "chain_type", # Optional: e.g., shape by chain type
                "r": True,  # Enable R-squared fitting
                "title": f"Func Group: {fg} (Chain Types: {chain_desc}, N={num_molecules}): Pred. Toxicity vs Total CF Groups"
            }
            experiments.append(experiment_config)
            print(f"Added experiment for Functional Group '{fg}' with {num_molecules} molecules.")
        else:
            print(
                f"Skipping Functional Group '{fg}' ({num_molecules} molecules), as it has only {unique_cf_lengths} unique CF length value(s).")
    return experiments
def AutoToxLengthScanByFunctionalGroupGlobal(df, base_conditions=None, min_molecules=20):
    """
    Generates an experiment configuration to plot toxicity vs. carbon chain length
    for all functional groups with more than a specified number of molecules,
    colored by functional group.
    Args:
        df (pd.DataFrame): The input dataframe.
        base_conditions (str, optional): A query string to filter the dataframe initially.
        min_molecules (int): Minimum number of molecules required for a functional group to be included. Defaults to 20.
    Returns:
        list: A list containing one experiment configuration dictionary.
    """
    experiments = []
    working_df = df.query(base_conditions) if base_conditions else df
    # Filter for non-null total_CF and functional_group
    filtered_df = working_df[
        (working_df['total_CF'].notna()) &
        (working_df['functional_group'].notna()) &
        (working_df['functional_group'] != '')  # Exclude empty strings
        ]
    grouped = filtered_df.groupby(['functional_group'])
    valid_groups = []
    for fg, group in grouped:
        num_molecules = len(group)
        if num_molecules > min_molecules:
            print(f"Including Functional Group '{fg}' with {num_molecules} molecules.")
            valid_groups.append(fg)
        else:
            print(f"Skipping Functional Group '{fg}', as it has only {num_molecules} molecules (<= {min_molecules}).")
    if not valid_groups:
        print("No functional groups met the minimum molecule count requirement.")
        return experiments
    # Filter the dataframe to include only molecules from valid groups
    condition_str = " | ".join([f"(functional_group == '{fg}')" for fg in valid_groups])
    final_condition = f"({condition_str})"
    if base_conditions:
        final_condition = f"({base_conditions}) & ({final_condition})"
    plot_df = working_df.query(final_condition)
    if plot_df.empty:
        print("Warning: No data found after applying filters.")
        return experiments
    print(f"Final plot will include {len(plot_df)} molecules from {len(valid_groups)} functional groups.")
    # Generate the single experiment configuration
    experiment_config = {
        "name": f"AllFuncGroups_Toxicity_vs_CF_Length_Global",
        "condition": final_condition, # Use the combined condition
        "analysis_type": "scatter_plot",
        "x_var": "total_CF",  # X-axis variable
        "y_var": "predicted_toxicity",  # Y-axis variable
        "hue_var": "functional_group",  # Color by functional group
        "style_var": "chain_type", # Optional: e.g., shape by chain type
        "r": False,  # Enable R-squared fitting if desired
        "title": f"All Included Func. Groups: Pred. Toxicity vs Total CF Groups (N={len(plot_df)}, Min Molecules Per Group > {min_molecules})"
    }
    experiments.append(experiment_config)
    return experiments

def AutoToxLengthScanByFluorinatedCarbonRatio(df, base_conditions=None, ratio_type='CF2_ratio'):
    """
    Generates an experiment configuration to plot predicted toxicity vs.
    a specific fluorinated carbon ratio (e.g., CF2_ratio, CF3_ratio, CFX_ratio).
    Args:
        df (pd.DataFrame): The input dataframe.
        base_conditions (str, optional): A query string to filter the dataframe initially.
        ratio_type (str): The type of ratio to use for X-axis. Options: 'CF2_ratio', 'CF3_ratio', 'CFX_ratio', 'Total_F_Ratio'.
                          Default is 'CF2_ratio'.
    Returns:
        list: A list containing one experiment configuration dictionary.
    """
    experiments = []
    working_df = df.query(base_conditions) if base_conditions else df
    # --- Calculate Ratios ---
    if 'carbon_chain_length' not in working_df.columns:
        print(f"Error: 'carbon_chain_length' column not found in data. Cannot calculate ratios.")
        return experiments
    total_carbon_col = working_df['CF2']+working_df['CF3']+working_df['CFX']
    # Calculate ratios (on a copy to avoid warnings if original df is a slice)
    working_df = working_df.copy()
    working_df['CF2_ratio'] = working_df['CF2'] / working_df[total_carbon_col]
    working_df['CF3_ratio'] = working_df['CF3'] / working_df[total_carbon_col]
    working_df['CFX_ratio'] = working_df['CFX'] / working_df[total_carbon_col]
    working_df['Total_F_Ratio'] = (working_df['CF2'] + working_df['CF3'] + working_df['CFX']) / working_df['carbon_chain_lengt']
    if ratio_type not in working_df.columns:
        print(f"Error: Ratio type '{ratio_type}' not found. Available options: {list(working_df[['CF2_ratio', 'CF3_ratio', 'CFX_ratio', 'Total_F_Ratio']].columns)}")
        return experiments
    # --- Only filter by the ratio column, NOT predicted_toxicity ---
    # The 'predicted_toxicity' column will be added later in run_sar_experiment
    filtered_df = working_df.dropna(subset=[ratio_type]) # Drop rows where the ratio itself is NaN
    if filtered_df.empty:
        print(f"Warning: No data found after calculating {ratio_type} and filtering for non-NaN values.")
        return experiments
    print(f"Calculated ratios. Found {len(filtered_df)} molecules with valid {ratio_type}.")
    # Generate the single experiment configuration
    experiment_config = {
        "name": f"Toxicity_vs_{ratio_type}_Analysis",
        "condition": base_conditions, # Use the base condition if provided, or None
        "analysis_type": "scatter_plot",
        "x_var": ratio_type,  # X-axis variable (the ratio)
        "y_var": "predicted_toxicity",  # Y-axis variable - will be added later
        "hue_var": "functional_group",  # Color by functional group (or another categorical column)
        "style_var": "chain_type", # Optional: e.g., shape by chain type
        "r": True,  # Enable R-squared fitting
        "title": f"Predicted Toxicity vs {ratio_type.replace('_', ' ')} (N={len(filtered_df)})"
    }
    experiments.append(experiment_config)
    return experiments

def AutoToxLengthScanByFluorinatedCarbonRatioDual(df, base_conditions=None, ratio_type='CF2_ratio'):
    """
    Generates an experiment configuration to plot predicted toxicity vs.
    both a specific fluorinated carbon absolute count and its ratio.
    """
    experiments = []
    working_df = df.query(base_conditions) if base_conditions else df
    if 'carbon_chain_length' not in working_df.columns:
        print(f"Error: 'carbon_chain_length' column not found in data. Cannot calculate ratios.")
        return experiments
    total_carbon_col = 'carbon_chain_length'
    working_df = working_df.copy()
    working_df['CF2_ratio'] = working_df['CF2'] / working_df[total_carbon_col]
    working_df['CF3_ratio'] = working_df['CF3'] / working_df[total_carbon_col]
    working_df['CFX_ratio'] = working_df['CFX'] / working_df[total_carbon_col]
    working_df['Total_F_Ratio'] = (working_df['CF2'] + working_df['CF3'] + working_df['CFX']) / working_df[total_carbon_col]
    if ratio_type not in working_df.columns:
        print(f"Error: Ratio type '{ratio_type}' not found. Available options: {list(working_df[['CF2_ratio', 'CF3_ratio', 'CFX_ratio', 'Total_F_Ratio']].columns)}")
        return experiments
    # Filter by the ratio column being non-NaN
    filtered_df = working_df.dropna(subset=[ratio_type])
    if filtered_df.empty:
        print(f"Warning: No data found after calculating {ratio_type} and filtering for non-NaN values.")
        return experiments
    print(f"Calculated ratios. Found {len(filtered_df)} molecules with valid {ratio_type}.")
    # Generate the single experiment configuration for dual plot
    experiment_config = {
        "name": f"Toxicity_vs_{ratio_type}_Dual_Analysis",
        "condition": base_conditions,
        "analysis_type": "dual_plot", # <-- Set to "dual_plot"
        "x_var": ratio_type,  # This will be used by execute_dual_plot to determine absolute and ratio vars
        "y_var": "predicted_toxicity",
        "hue_var": "functional_group",
        "style_var": "chain_type", # Optional: e.g., shape by chain type
        "r": True,
        "title": f"Predicted Toxicity vs {ratio_type.replace('_', ' ')} and Absolute Count (N={len(filtered_df)})"
    }
    experiments.append(experiment_config)
    return experiments

# --- 6.3. 定义专门用于绘制特定类别组合对比图的函数 ---
def AutoToxLengthScanBySpecificCombinations(df, target_combinations, base_conditions=None):
    """
    Generates a single experiment configuration for plotting multiple specific class combinations together.
    The plot will use 'functional_group' as the hue variable.
    Args:
        df (pd.DataFrame): The input dataframe with all data.
        target_combinations (list of tuples): List of (first_class, second_class) tuples to include in the plot.
        base_conditions (str, optional): Additional Pandas query string to pre-filter the data before applying the main condition.
    Returns:
        list: A list containing one experiment config dict.
    """
    experiments = []
    # Apply base conditions if provided
    working_df = df.query(base_conditions) if base_conditions else df
    # Create a condition string that matches any of the target combinations
    conditions_list = []
    for fc, sc in target_combinations:
        # Escape single quotes if they exist in the class names
        fc_escaped = fc.replace("'", "\\'")
        sc_escaped = sc.replace("'", "\\'")
        conditions_list.append(f"(first_class == '{fc_escaped}' and second_class == '{sc_escaped}')")
    combined_condition = " or ".join(conditions_list)
    full_query = f"({combined_condition})"
    if base_conditions:
        full_query = f"({base_conditions}) and ({combined_condition})"
    # Check how many molecules match this condition
    subset = working_df.query(full_query)
    num_molecules = len(subset)
    print(f"为特定组合分析准备数据: {num_molecules} 个分子匹配条件 '{full_query}'.")
    if num_molecules > 0:
        # Generate a descriptive name for the experiment
        # Use the first few combinations for the name to avoid it being too long
        name_parts = [f"{fc.replace(' ', '_')}_vs_{sc.replace(' ', '_')}" for fc, sc in target_combinations[:3]]
        name_suffix = "_etc" if len(target_combinations) > 3 else ""
        experiment_name = f"{'_and_'.join(name_parts)}{name_suffix}_Toxicity_vs_CF_Length"
        experiment_config = {
            "name": experiment_name,
            "condition": full_query, # Use the full combined condition
            "analysis_type": "scatter_plot",
            "x_var": "total_CF",
            "y_var": "predicted_toxicity",
            "hue_var": "second_class", # 这是关键：用官能团着色
            "style_var": "chain_type", # Optional: e.g., shape by chain type
            "r": False, # 启用 R² 拟合线
            "title": f"Specific Combinations ({len(target_combinations)} groups, N={num_molecules}): Predicted Toxicity vs Total CF Groups"
        }
        experiments.append(experiment_config)
        print(f"成功为 {len(target_combinations)} 个特定类别组合生成了实验配置。")
    else:
        print("警告: 没有分子匹配指定的类别组合条件。")
    return experiments
# --- 6.4. 定义要绘制的具体类别组合列表 ---
# 根据您提供的表格中黄标高亮的部分，我们提取出对应的 (first_class, second_class) 组合
TARGET_COMBINATIONS = [
    ("PFAA precursors", "HFCs"),
    ("PFAA precursors", "PFAIs"),
    ("PFAA precursors", "PFAenes"),
    ("PFAA precursors", "PolyFACs"),
    ("PFAA precursors", "SFAenes"),
    ("PFAAs", "PFCA-ester derivatives"),
    ("Polyfluoroalkyl acids", "PolyFCAs"),
]
# --- 6. 主程序 ---
if __name__ == "__main__":
    # 1. 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    # 2. 加载模型
    model = load_pretrained_model(PRETRAINED_MODEL_PATH, new_prediction_tasks=1)
    model = model.to(device)
    print("模型加载完成。")
    # --- 3. 定义所有实验配置的生成函数列表 (使用字典) ---
    # 这样可以轻松添加或移除不同类型的分析，且参数更灵活
    experiment_generators = [
        # {
        #     "description": "按官能团扫描 (Function Group Scan)",
        #     "function": AutoToxLengthScanByFunctionGroup,
        #     "kwargs": {"base_conditions": None}
        # },
        {
            "description": "全局官能团毒性-碳链长度分析",
            "function": AutoToxLengthScanByCategory,
            "kwargs": {"base_conditions": None}
        },
        # {
        #     "description": "按类别扫描 (Class Scan, Straight Chains)",
        #     "function": AutoToxLengthScanByCategory,
        #     "kwargs": {"base_conditions": "(chain_type == 'straight')"}
        # },
        # {
        #     "description": "特定类别组合对比图 (PFAA precursors & others, hue=Functional Group)",
        #     "function": AutoToxLengthScanBySpecificCombinations,
        #     "kwargs": {"target_combinations": TARGET_COMBINATIONS, "base_conditions": None}  # 可以在这里添加 base_conditions
        # },
        # # --- 新增：按含氟碳比例扫描 (双子图) ---
        # {
        #     "description": "毒性 vs CF2 比例和绝对值分析 (双子图)",
        #     "function": AutoToxLengthScanByFluorinatedCarbonRatioDual, # Use the new function
        #     "kwargs": {"base_conditions": None, "ratio_type": 'CF2_ratio'}
        # },
        # {
        #     "description": "毒性 vs CF3 比例和绝对值分析 (双子图)",
        #     "function": AutoToxLengthScanByFluorinatedCarbonRatioDual,
        #     "kwargs": {"base_conditions": None, "ratio_type": 'CF3_ratio'}
        # },
        # {
        #     "description": "毒性 vs CFX 比例和绝对值分析 (双子图)",
        #     "function": AutoToxLengthScanByFluorinatedCarbonRatioDual,
        #     "kwargs": {"base_conditions": None, "ratio_type": 'CFX_ratio'}
        # },
        # {
        #     "description": "毒性 vs 总含氟碳比例和绝对值分析 (双子图)",
        #     "function": AutoToxLengthScanByFluorinatedCarbonRatioDual,
        #     "kwargs": {"base_conditions": None, "ratio_type": 'Total_F_Ratio'}
        # },
        # 示例：其他分析
        # {
        #     "description": "其他分析1",
        #     "function": other_analysis_func1,
        #     "kwargs": {"base_conditions": "some_condition"}
        # },
        # {
        #     "description": "其他分析2",
        #     "function": other_analysis_func2,
        #     "kwargs": {"base_conditions": None}
        # },
    ]
    # --- 4. 通用执行循环 ---
    all_experiments_run = 0
    for exp_gen_info in experiment_generators:
        description = exp_gen_info["description"]
        generator_func = exp_gen_info["function"]
        kwargs = exp_gen_info["kwargs"] # Get the keyword arguments
        print(f"--- 开始: {description} ---")
        # 调用相应的生成函数创建实验配置, 传入 kwargs
        experiment_configs = generator_func(df_enhanced, **kwargs) # Use ** to unpack the dictionary
        if experiment_configs:
            print(f"为 '{description}' 生成了 {len(experiment_configs)} 个符合条件的实验配置。")
            for exp_cfg in experiment_configs:
                # 通用执行函数
                run_sar_experiment(
                    df=df_enhanced,
                    model=model,
                    device=device,
                    experiment_name=exp_cfg["name"],
                    query_condition=exp_cfg["condition"],
                    exp_config=exp_cfg,  # Pass the config so visualization can use hue_var and r
                    smiles_col='SMILES'
                )
                all_experiments_run += 1
        else:
            print(f"'{description}' 没有找到符合条件的实验配置。")
    print(f"--- 所有实验完成，共执行了 {all_experiments_run} 个实验。 ---")