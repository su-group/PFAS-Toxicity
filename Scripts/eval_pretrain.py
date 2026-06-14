#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练模型评估模块
直接加载已训练好的预训练模型，在 Train/Val/Test 三集上进行完整评估
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from config import get_model_save_path
from dataset.molecule_dataset import MoleculeDataset
from models.gnn_model import MoleculeGNN, UncertaintyLoss
from models.gat_model import MoleculeGAT
from models.gin_model import MoleculeGIN
from models.jknet_model import MoleculeJKNet
from models.transformer_model import MoleculeGraphTransformer

# ==================== 配置区（与预训练完全一致） ====================
DATA_CONFIG = {
    "data_dir": r"F:\GNN-pro\data\pretrain",
    "csv_file": "ldtoxdb-nof.csv",
    "smiles_col": "SMILES",
    "label_col": "NeglogLD50",
    "batch_size": 32,
    "test_size": 0.1,
    "val_size": 0.1,
    "random_state": 42,
    "normalize_labels": True,
    "max_atoms": 512
}

MODEL_CONFIG = {
    "model_type": "GIN",
    "supported_types": ["GNN", "GAT", "GIN", "Transformer", "JKNet"],
    "params": {
        "hidden_dim": 256,
        "num_layers": 8,
        "dropout": 0.2,
        "prediction_tasks": 1
    },
    "type_specific_params": {
        "GNN": {"model_type": "GCN"},
        "GAT": {"num_heads": 4},
        "JKNet": {"jk_mode": "cat"},
        "Transformer": {"num_heads": 8}
    }
}

TRAINING_CONFIG = {
    "epochs": 1000,
    "learning_rate": 0.001,
    "weight_decay": 1e-05,
    "checkpoint_interval": 10,
    "device": "cuda"
}

# 指定要评估的模型文件（可改为 checkpoint 或 final 模型）
MODEL_TO_EVAL = get_model_save_path("F:\GNN-pro\scripts\outputs\GIN_20260107-094723\pretrained_best_model.pth")


# ==================== 核心函数 ====================
def create_model(config):
    """根据配置创建模型（与预训练脚本完全一致）"""
    model_type = config['model_type']
    model_params = {
        'hidden_dim': config['params']['hidden_dim'],
        'num_layers': config['params']['num_layers'],
        'dropout': config['params']['dropout'],
        'prediction_tasks': config['params']['prediction_tasks'],
        'input_dim': config.get('input_dim', 10)
    }

    if model_type == "GNN":
        model_params['model_type'] = config['type_specific_params']['GNN']['model_type']
        model = MoleculeGNN(**model_params)
    elif model_type == "GAT":
        model_params['num_heads'] = config['type_specific_params']['GAT']['num_heads']
        model = MoleculeGAT(**model_params)
    elif model_type == "GIN":
        model = MoleculeGIN(**model_params)
    elif model_type == "JKNet":
        model_params['jk_mode'] = config['type_specific_params']['JKNet']['jk_mode']
        model = MoleculeJKNet(**model_params)
    elif model_type == "Transformer":
        model_params['num_heads'] = config['type_specific_params']['Transformer']['num_heads']
        model = MoleculeGraphTransformer(**model_params)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    return model


def load_data(config):
    """加载数据并返回三集加载器（与预训练脚本完全一致）"""
    dataset = MoleculeDataset(
        root=config['data_dir'],
        filename=config['csv_file'],
        smiles_col=config['smiles_col'],
        label_col=config['label_col']
    )

    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(
        indices, test_size=config['test_size'] + config['val_size'],
        random_state=config['random_state']
    )
    val_size_relative = config['val_size'] / (config['test_size'] + config['val_size'])
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=val_size_relative, random_state=config['random_state']
    )

    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    input_dim = dataset[0].x.shape[1] if len(dataset) > 0 else 10
    return train_loader, val_loader, test_loader, input_dim, dataset


def load_norm_params(data_dir):
    """安全加载标准化参数（用于反标准化预测值）"""
    norm_path = os.path.join(data_dir, 'processed', 'norm_params.npy')
    if os.path.exists(norm_path):
        try:
            params = np.load(norm_path, allow_pickle=True).item()
            return {'mean': float(params['mean']), 'std': float(params['std'])}
        except Exception as e:
            print(f"⚠️ 加载标准化参数失败: {e}，将使用标准化值计算指标")
    return None


def evaluate_split(model, loader, device, norm_params):
    """评估单个数据集"""
    model.eval()
    criterion = UncertaintyLoss()

    preds_norm, targets_norm = [], []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data in tqdm(loader, desc="评估", leave=False):
            data = data.to(device)
            output, uncertainty = model(data.x, data.edge_index, data.batch)
            loss = criterion(output, data.y, uncertainty)

            total_loss += loss.item() * data.num_graphs
            total_samples += data.num_graphs
            preds_norm.append(output.cpu().numpy().ravel())
            targets_norm.append(data.y.cpu().numpy().ravel())

    preds_norm = np.concatenate(preds_norm)
    targets_norm = np.concatenate(targets_norm)
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')

    # 反标准化恢复原始物理尺度
    if norm_params:
        preds = preds_norm * norm_params['std'] + norm_params['mean']
        targets = targets_norm * norm_params['std'] + norm_params['mean']
    else:
        preds, targets = preds_norm, targets_norm

    # 计算回归指标
    metrics = {
        'Loss': avg_loss,
        'Samples': total_samples,
        'R2': r2_score(targets, preds),
        'RMSE': np.sqrt(mean_squared_error(targets, preds)),
        'MAE': mean_absolute_error(targets, preds)
    }
    return metrics


def print_eval_table(results_dict):
    """打印论文级评估表格"""
    print("\n" + "=" * 90)
    print(f"{'Pretrained Model Evaluation Results':^90}")
    print("=" * 90)
    print(f"{'Split':<12} | {'Samples':<8} | {'Loss':<10} | {'R²':<10} | {'RMSE':<10} | {'MAE':<10}")
    print("-" * 90)

    for split_name, metrics in results_dict.items():
        print(f"{split_name:<12} | {metrics['Samples']:<8} | {metrics['Loss']:<10.4f} | "
              f"{metrics['R2']:<10.4f} | {metrics['RMSE']:<10.4f} | {metrics['MAE']:<10.4f}")
    print("=" * 90 + "\n")


# ==================== 主评估流程 ====================
def main():
    print("=" * 80)
    print(" 开始预训练模型评估")
    print("=" * 80)

    if not os.path.exists(MODEL_TO_EVAL):
        print(f"❌ 错误: 模型文件不存在 → {MODEL_TO_EVAL}")
        print(" 提示: 请确认预训练是否已完成，或修改 MODEL_TO_EVAL 路径")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 使用设备: {device}")

    # 1. 加载数据
    print("\n📂 加载数据集...")
    train_loader, val_loader, test_loader, input_dim, dataset = load_data(DATA_CONFIG)
    print(
        f"   训练集: {len(train_loader.dataset)} | 验证集: {len(val_loader.dataset)} | 测试集: {len(test_loader.dataset)}")

    # 2. 加载标准化参数
    norm_params = load_norm_params(DATA_CONFIG['data_dir'])
    if norm_params:
        print(f"   ✅ 加载标准化参数: mean={norm_params['mean']:.4f}, std={norm_params['std']:.4f}")

    # 3. 构建并加载模型
    print(f"\n🧠 加载预训练模型: {os.path.basename(MODEL_TO_EVAL)}")
    MODEL_CONFIG['input_dim'] = input_dim
    model = create_model(MODEL_CONFIG)

    checkpoint = torch.load(MODEL_TO_EVAL, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    print("✅ 模型加载成功")

    # 4. 评估三集
    print("\n🔍 开始全面评估...")
    results = {}

    split_loaders = {
        'Train': train_loader,
        'Validation': val_loader,
        'Test': test_loader
    }

    for name, loader in split_loaders.items():
        print(f"\n  评估 {name} 集...")
        results[name] = evaluate_split(model, loader, device, norm_params)
        print(
            f"    ✓ Loss={results[name]['Loss']:.4f} | R²={results[name]['R2']:.4f} | RMSE={results[name]['RMSE']:.4f} | MAE={results[name]['MAE']:.4f}")

    # 5. 输出结果
    print_eval_table(results)

    # 6. 保存 CSV 报告
    report_dir = os.path.join(os.path.dirname(MODEL_TO_EVAL), 'evaluation_report.csv')
    pd.DataFrame(results).T.to_csv(report_dir, float_format='%.4f')
    print(f" 评估报告已保存至: {report_dir}")
    print("\n✅ 评估完成！")


if __name__ == "__main__":
    main()