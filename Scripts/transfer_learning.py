#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型迁移学习模块

该模块提供模型迁移学习功能，加载预训练模型，根据新任务调整模型结构，
并使用新数据对模型进行微调。
"""

import os
import sys
import torch
import torch.nn as nn
from datetime import datetime
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
import json

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from config import get_model_save_path
from dataset.molecule_dataset import MoleculeDataset
from models.gnn_model import MoleculeGNN, UncertaintyLoss
from models.gat_model import MoleculeGAT
from models.gin_model import MoleculeGIN
from models.jknet_model import MoleculeJKNet
from utils import save_model, create_tensorboard_writer, log_config_to_tensorboard, \
                  log_losses_to_tensorboard, close_tensorboard_writer
from sklearn.model_selection import train_test_split

# --- 配置 ---
DATA_CONFIG = {
    "data_dir": "F:\GNN-pro\data\transfer",
    "csv_file": "PFAS_id50_step3.csv",
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
    "epochs": 500,
    "learning_rate": 0.001,
    "weight_decay": 1e-05,
    "checkpoint_interval": 10,
    "device": "cuda"
}

PRETRAIN_CONFIG = {
    'data_config': DATA_CONFIG,
    'model_config': MODEL_CONFIG,
    'training_config': TRAINING_CONFIG,
    'save_config': {
        'checkpoint_interval': 10,
        'save_best_model': True,
        'save_final_model': True
    }
}

TRANSFER_CONFIG = {
    'pretrained_model_path': get_model_save_path(r"F:\GNN-pro\scripts\outputs\GIN_20260107-094723\pretrained_best_model.pth"),
    'new_data_config': {
        "data_dir": r"F:\GNN-pro\data\transfer",
        "csv_file": "PFAS_id50_step3.csv",
        "smiles_col": "SMILES",
        "label_col": "NeglogLD50",
        "batch_size": 32,
        "test_size": 0.1,
        "val_size": 0.1,
        "random_state": 42,
        "normalize_labels": True,
        "max_atoms": 512
    },
    'new_model_config': {
        'params': {
            'prediction_tasks': 1,
        },
    },
    'transfer_training_config': {
        'epochs': 500,
        'learning_rate': 1e-4,
        'weight_decay': 1e-6,
        'freeze_base': False,
    },
    'save_config': {
        'checkpoint_interval': 10,
        'save_best_model': True,
        'save_final_model': True,
    }
}


def load_pretrained_model(model_path, new_prediction_tasks):
    """
    加载预训练模型并调整输出层以适应新任务。
    不依赖 config.json，直接使用 MODEL_CONFIG 和合理默认值。
    """
    print(f"正在加载预训练模型: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    loaded_model_state = checkpoint.get('model_state_dict', checkpoint)

    # 使用 MODEL_CONFIG 中的参数重建模型结构
    original_model_type = MODEL_CONFIG["model_type"]  # "GNN"
    original_prediction_tasks = MODEL_CONFIG["params"]["prediction_tasks"]
    input_dim = 120  # 分子图常用原子特征维度（如 RDKit + one-hot）
    hidden_dim = MODEL_CONFIG["params"]["hidden_dim"]
    num_layers = MODEL_CONFIG["params"]["num_layers"]
    dropout = MODEL_CONFIG["params"]["dropout"]
    type_specific_params = MODEL_CONFIG.get("type_specific_params", {})

    print(f"原始模型类型: {original_model_type}, 原始预测任务数: {original_prediction_tasks}, 新预测任务数: {new_prediction_tasks}")
    print(f"使用预训练模型的结构参数: input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}")

    model_params = {
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'prediction_tasks': original_prediction_tasks,
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

    # 加载权重（允许输出层不匹配）
    model.load_state_dict(loaded_model_state, strict=False)

    # 替换预测层（如果任务数不同）
    if hasattr(model, 'predictor'):
        if original_prediction_tasks != new_prediction_tasks:
            print(f"替换预测层: 从 {original_prediction_tasks} -> {new_prediction_tasks}")
            last_layer_input_features = hidden_dim
            model.predictor = nn.Linear(last_layer_input_features, new_prediction_tasks)
        else:
            print(f"新旧任务数相同 ({new_prediction_tasks})，保留原始预测层权重。")
    else:
        print("警告: 未找到名为 'predictor' 的预测层。请检查模型结构并手动调整输出层。")

    model.prediction_tasks = new_prediction_tasks
    return model, None  # 第二个返回值不再使用


def load_data(config):
    dataset = MoleculeDataset(
        root=config['data_dir'],
        filename=config['csv_file'],
        smiles_col=config['smiles_col'],
        label_col=config['label_col']
    )

    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=config['test_size'] + config['val_size'],
        random_state=config['random_state']
    )
    val_size_relative = config['val_size'] / (config['test_size'] + config['val_size'])
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=val_size_relative,
        random_state=config['random_state']
    )

    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    input_dim = dataset[0].x.shape[1] if len(dataset) > 0 else 119
    return train_loader, val_loader, test_loader, input_dim


def transfer_learn():
    print("开始模型迁移学习...")

    pretrained_path = TRANSFER_CONFIG['pretrained_model_path']
    new_data_config = TRANSFER_CONFIG['new_data_config']
    new_model_config = TRANSFER_CONFIG['new_model_config']
    transfer_training_config = TRANSFER_CONFIG['transfer_training_config']
    save_config = TRANSFER_CONFIG['save_config']

    new_prediction_tasks = new_model_config['params']['prediction_tasks']

    if not os.path.exists(pretrained_path):
        print(f"错误: 预训练模型文件不存在: {pretrained_path}")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载并调整模型
    print("加载预训练模型并调整结构...")
    model, _ = load_pretrained_model(pretrained_path, new_prediction_tasks)  # 忽略第二个返回值
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"调整后模型可训练参数数量: {total_params}")

    # 冻结基础层（如果需要）
    if transfer_training_config.get('freeze_base', False):
        print("冻结基础模型权重，只训练新的预测层...")
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, 'predictor'):
            for param in model.predictor.parameters():
                param.requires_grad = True
        print(f"现在可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 加载数据
    print("加载新任务数据...")
    train_loader, val_loader, test_loader, _ = load_data(new_data_config)
    print(f"新训练集大小: {len(train_loader.dataset)}")
    print(f"新验证集大小: {len(val_loader.dataset)}")
    print(f"新测试集大小: {len(test_loader.dataset)}")

    # 损失与优化器
    criterion = UncertaintyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=transfer_training_config['learning_rate'],
        weight_decay=transfer_training_config['weight_decay']
    )

    # TensorBoard
    writer = create_tensorboard_writer()
    log_config_to_tensorboard(writer, new_data_config, 'Transfer_Data')
    log_config_to_tensorboard(writer, new_model_config, 'Transfer_Model')
    log_config_to_tensorboard(writer, transfer_training_config, 'Transfer_Training')

    print("开始迁移学习训练...")

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    epochs = transfer_training_config['epochs']

    for epoch in tqdm(range(epochs), position=0, leave=True):
        # 训练
        model.train()
        train_loss = 0.0
        train_samples = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output, uncertainty = model(data.x, data.edge_index, data.batch)
            loss = criterion(output, data.y, uncertainty)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.num_graphs
            train_samples += data.num_graphs
        avg_train_loss = train_loss / train_samples
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output, uncertainty = model(data.x, data.edge_index, data.batch)
                loss = criterion(output, data.y, uncertainty)
                val_loss += loss.item() * data.num_graphs
                val_samples += data.num_graphs
        avg_val_loss = val_loss / val_samples
        val_losses.append(avg_val_loss)

        log_losses_to_tensorboard(writer, avg_train_loss, avg_val_loss, epoch)

        # 保存最佳模型
        if avg_val_loss < best_val_loss and save_config['save_best_model']:
            best_val_loss = avg_val_loss
            best_model_path = get_model_save_path("transfer_learned_best_model.pth")
            save_model(
                model,
                new_model_config.get('model_type', 'Unknown'),
                new_model_config,
                best_model_path,
                epoch,
                avg_train_loss,
                avg_val_loss
            )
            print(f"在第 {epoch+1} 轮保存了最佳迁移模型，验证损失: {avg_val_loss:.4f}")

        # 保存检查点
        if (epoch + 1) % save_config['checkpoint_interval'] == 0:
            checkpoint_path = get_model_save_path(f"transfer_checkpoint_epoch_{epoch+1}.pth")
            save_model(
                model,
                new_model_config.get('model_type', 'Unknown'),
                new_model_config,
                checkpoint_path,
                epoch,
                avg_train_loss,
                avg_val_loss,
                optimizer.state_dict()
            )

    # 保存最终模型
    if save_config['save_final_model']:
        final_model_path = get_model_save_path("transfer_learned_final_model.pth")
        save_model(
            model,
            new_model_config.get('model_type', 'Unknown'),
            new_model_config,
            final_model_path,
            epochs - 1,
            train_losses,
            val_losses
        )
        print(f"最终迁移模型已保存到: {final_model_path}")

    close_tensorboard_writer(writer)
    print("迁移学习完成!")

    # 测试评估
    model.eval()
    test_loss = 0.0
    test_samples = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output, uncertainty = model(data.x, data.edge_index, data.batch)
            loss = criterion(output, data.y, uncertainty)
            test_loss += loss.item() * data.num_graphs
            test_samples += data.num_graphs
    avg_test_loss = test_loss / test_samples
    print(f"新任务测试集损失: {avg_test_loss:.4f}")

    return model, train_losses, val_losses


if __name__ == "__main__":
    transfer_learn()