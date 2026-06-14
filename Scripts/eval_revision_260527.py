#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整数据集评估模块（Train/Val/Test）- 类型安全修复版
响应审稿意见：同时报告训练集、验证集、测试集的性能指标
已修复：列表与Tensor类型冲突、反标准化崩溃、损失函数不匹配
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_model_save_path
from dataset.molecule_dataset import MoleculeDataset
from models.gnn_model import MoleculeGNN
from models.gat_model import MoleculeGAT
from models.gin_model import MoleculeGIN
from models.jknet_model import MoleculeJKNet
from models.transformer_model import MoleculeGraphTransformer

# ==================== 配置区 ====================
EVAL_CONFIG = {
    'model_path': get_model_save_path(
        r"F:\GNN-pro\scripts\outputs\GNN\GNN_cf2_Start_task_20260528-190851\transfer_learned_best_model.pth"

    ),
    'model_type': 'GNN',
    'model_params': {
        'hidden_dim': 256,
        'num_layers': 8,
        'dropout': 0.2,
        'prediction_tasks': 1,
    },
    'data_config': {
        'data_dir': r'F:\GNN-pro\data\cf2_data_start',
        'train_csv': r'train\train_filled.csv',
        'val_csv': r'val\val_filled.csv',
        'test_csv': r'test\test_filled.csv',
        'smiles_col': 'SMI',
        'label_col': 'NeglogLD50',
        'batch_size': 32,
    },
    'save_results': True,
    'output_dir': r'F:\GNN-pro\data\cf2_data',
    'method_name': 'GNN-TL-Start',
}


def load_model(model_path, model_type, model_params, input_dim):
    """加载模型"""
    model_classes = {
        "GNN": MoleculeGNN,
        "GAT": MoleculeGAT,
        "GIN": MoleculeGIN,
        "JKNet": MoleculeJKNet,
        "Transformer": MoleculeGraphTransformer
    }
    if model_type not in model_classes:
        raise ValueError(f"不支持的模型类型: {model_type}")

    params = {**model_params, 'input_dim': input_dim}
    model = model_classes[model_type](**params)

    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_all_datasets(data_config):
    """加载训练、验证、测试三个数据集"""
    print("\n📂 加载数据集...")
    datasets, loaders = {}, {}
    data_dir = data_config['data_dir']

    for split in ['train', 'val', 'test']:
        csv_file = data_config[f'{split}_csv']
        csv_path = os.path.join(data_dir, csv_file)
        if not os.path.exists(csv_path):
            print(f"  ⚠️  警告: {csv_path} 不存在，跳过{split}集")
            continue

        dataset = MoleculeDataset(
            root=os.path.join(data_dir, split),
            filename=csv_file,
            smiles_col=data_config['smiles_col'],
            label_col=data_config['label_col']
        )
        loader = DataLoader(dataset, batch_size=data_config['batch_size'], shuffle=False)
        datasets[split], loaders[split] = dataset, loader
        print(f"  ✓ {split.upper():6} | 样本数: {len(dataset):4} | 特征维度: {dataset[0].x.shape[1]}")

    return datasets, loaders


def evaluate_split(model, loader, device, data_dir):
    """评估单个数据集（完全类型安全，不依赖 dataset 方法）"""
    model.eval()
    predictions, targets = [], []
    total_loss = 0.0
    total_samples = 0

    # 1. 安全加载标准化参数（从文件读取，不依赖对象属性）
    # norm_path = os.path.join(data_dir, 'train', 'processed', 'norm_params.npy')
    # if os.path.exists(norm_path):
    #     try:
    #         params = np.load(norm_path, allow_pickle=True).item()
    #         # 确保转成 Python float，防止 Tensor 类型冲突
    #         mean_val = float(params['mean'])
    #         std_val = float(params['std'])
    #     except Exception as e:
    #         print(f"  ⚠️ 加载参数失败: {e}，使用兜底值")
    #         mean_val, std_val = 0.1512, 0.1950
    # else:
    #     mean_val, std_val = 0.1512, 0.1950

    # 使用与训练一致的 MSE Loss 进行评估
    criterion = nn.MSELoss()

    # 2. 推理循环
    with torch.no_grad():
        for data in tqdm(loader, desc=f"评估", leave=False):
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)

            # 兼容 tuple 输出 (prediction, uncertainty) 或 tensor 输出
            prediction = output[0] if isinstance(output, tuple) else output

            # 计算 Loss (prediction 形状为 [B, 1]，data.y 为 [B]，需 squeeze)
            loss = criterion(prediction.squeeze(-1), data.y)
            total_loss += loss.item() * data.num_graphs
            total_samples += data.num_graphs

            # 收集结果（直接转为 numpy float32，避免列表类型冲突）
            predictions.extend(prediction.cpu().numpy().ravel())
            targets.extend(data.y.cpu().numpy().ravel())

    # 3. 类型转换 & 反标准化（纯 Numpy 运算，绝对安全）
    # ✅ 手动安全反标准化（彻底避开 dataset 方法的类型冲突）
    # norm_path = os.path.join(EVAL_CONFIG['data_config']['data_dir'], 'train', 'processed', 'norm_params.npy')
    # if os.path.exists(norm_path):
    #     params = np.load(norm_path, allow_pickle=True).item()
    #     mean_val = float(params['mean'])
    #     std_val = float(params['std'])
    # else:
    mean_val, std_val = 2.9348, 1.0857

    preds = np.array(predictions, dtype=np.float32) * std_val + mean_val  # 预测值反标准化到原始空间
    targets = np.array(targets, dtype=np.float32) * std_val + mean_val  # targets也反标准化到原始空间（真实值）
    print("pre",preds)
    print("tar",targets)

    # ✅ 手动反标准化：不再调用 dataset.inverse_normalize_labels
    # predictions = np.array(predictions, dtype=np.float32)
    # targets = np.array(targets, dtype=np.float32)
    # predictions = predictions * std_val + mean_val
    # print("pre"+ predictions)
    # targets = targets * std_val + mean_val
    # print("tar"+targets)
    # predictions = np.clip(predictions, 0.0, 1.0)  # 约束到百分数范围

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    return preds, targets, avg_loss


def calculate_split_metrics(y_true, y_pred, loss):
    """计算指标"""
    return {
        'Loss': loss,
        'Samples': len(y_true),
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
    }


def print_comprehensive_table(all_results, method_name="Model"):
    """打印表格"""
    print("\n" + "=" * 90)
    print(f"{'Table 1. Performance comparison across Train/Validation/Test sets':^90}")
    print("=" * 90)
    print(f"{'Method':<18} | {'Split':<10} | {'Samples':<8} | {'R²':<10} | {'RMSE':<10} | {'MAE':<10}")
    print("-" * 90)

    for split in ['train', 'val', 'test']:
        if split in all_results:
            m = all_results[split]
            split_name = split.upper() if split != 'val' else 'Validation'
            print(f"{method_name:<18} | {split_name:<10} | {m['Samples']:<8} | "
                  f"{m['R2']:<10.4f} | {m['RMSE']:<10.4f} | {m['MAE']:<10.4f}")
    print("=" * 90)

    if 'train' in all_results and 'test' in all_results:
        gap = all_results['train']['R2'] - all_results['test']['R2']
        print(f"\n📊 过拟合分析 (Train R² - Test R² = {gap:.4f})")
        if gap < 0.05:
            print("  ✅ 泛化能力优秀")
        elif gap < 0.15:
            print("  ✅ 泛化能力良好")
        elif gap < 0.30:
            print("  ️ 轻微过拟合")
        else:
            print("   严重过拟合")
    print("=" * 90 + "\n")


def save_results_to_csv(all_results, output_path, method_name):
    """保存 CSV"""
    data = []
    for split in ['train', 'val', 'test']:
        if split in all_results:
            m = all_results[split]
            data.append({
                'Method': method_name,
                'Split': split.capitalize(),
                'Samples': m['Samples'],
                'R2': m['R2'],
                'RMSE': m['RMSE'],
                'MAE': m['MAE'],
                'Loss': m['Loss']
            })
    pd.DataFrame(data).to_csv(output_path, index=False, float_format='%.4f')
    print(f" 结果已保存至: {output_path}")


def main():
    print("=" * 90)
    print(f"{'完整数据集评估系统 (Train/Validation/Test)':^90}")
    print(f"{'响应审稿意见：同时报告训练/验证/测试集指标':^90}")
    print("=" * 90)

    model_path = EVAL_CONFIG['model_path']
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在 → {model_path}");
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 使用设备: {device}")

    datasets, loaders = load_all_datasets(EVAL_CONFIG['data_config'])
    if not loaders:
        print("❌ 错误: 未找到任何数据集");
        sys.exit(1)

    input_dim = datasets[list(datasets.keys())[0]][0].x.shape[1]
    print(f"\n 加载模型: {os.path.basename(model_path)}")
    model = load_model(model_path, EVAL_CONFIG['model_type'], EVAL_CONFIG['model_params'], input_dim)
    model = model.to(device)
    print("✅ 模型加载成功")

    data_dir = EVAL_CONFIG['data_config']['data_dir']
    print("\n🔍 开始评估所有数据集...")

    all_results = {}
    for split, loader in loaders.items():
        print(f"\n  评估 {split.upper()} 集...")
        preds, targets, loss = evaluate_split(model, loader, device, data_dir)

        all_results[split] = calculate_split_metrics(targets, preds, loss)
        print(
            f"    ✓ R²={all_results[split]['R2']:.4f}, RMSE={all_results[split]['RMSE']:.4f}, MAE={all_results[split]['MAE']:.4f}")

    print_comprehensive_table(all_results, EVAL_CONFIG['method_name'])

    if EVAL_CONFIG['save_results']:
        os.makedirs(EVAL_CONFIG['output_dir'], exist_ok=True)
        save_results_to_csv(all_results,
                            os.path.join(EVAL_CONFIG['output_dir'],
                                         f"table1_{EVAL_CONFIG['method_name']}_all_splits.csv"),
                            EVAL_CONFIG['method_name'])

    print("✅ 评估完成！")
    return all_results


if __name__ == "__main__":
    main()