#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型预训练模块

该模块提供模型预训练功能，包括数据加载、模型配置、训练过程和结果保存
"""

import os
import sys
import torch
import json
from datetime import datetime
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
# from config import config_manager, get_data_config, get_model_config, get_training_config
from config import get_model_save_path
from dataset.molecule_dataset import MoleculeDataset
from models.gnn_model import MoleculeGNN, UncertaintyLoss
from models.gat_model import MoleculeGAT
from models.gin_model import MoleculeGIN
from models.jknet_model import MoleculeJKNet
from models.transformer_model import MoleculeGraphTransformer
from utils import save_model, create_tensorboard_writer, log_config_to_tensorboard, \
                  log_losses_to_tensorboard, close_tensorboard_writer
from sklearn.model_selection import train_test_split


# 获取配置
DATA_CONFIG = {
        "data_dir": "F:\GNN-pro\data\pretrain",
        "csv_file": "ldtoxdb-nof.csv",
        "smiles_col": "SMILES",
        "label_col": "NeglogLD50",
        "batch_size": 32,
        "test_size": 0.1,
        "val_size": 0.1,
        "random_state": 42,
        "normalize_labels": True,
        "max_atoms": 512}

MODEL_CONFIG = {
    "model_type": "GIN",
    "supported_types": ["GNN", "GAT", "GIN", "Transformer", "JKNet"],  # 列表形式
    "params": {
        "hidden_dim": 256,
        "num_layers": 8,
        "dropout": 0.2,
        "prediction_tasks": 1
    },
    "type_specific_params": {
        "GNN": {
            "model_type": "GCN"  # 或其他GNN变体类型
        },
        "GAT": {
            "num_heads": 4
        },
        "JKNet": {
            "jk_mode": "cat"  # 或 "max", "lstm" 等
        },
        "Transformer": {
            "num_heads": 8
        }
    }
}

TRAINING_CONFIG = {
        "epochs": 1000,
        "learning_rate": 0.001,
        "weight_decay": 1e-05,
        "checkpoint_interval": 10,
         "device": "cuda"
    }

# 配置字典
PRETRAIN_CONFIG = {
    # 数据配置
    'data_config': DATA_CONFIG,
    
    # 模型配置
    'model_config': MODEL_CONFIG,
    
    # 训练配置
    'training_config': TRAINING_CONFIG,
    
    # 存储配置
    'save_config': {
        'checkpoint_interval':  10,
        'save_best_model': True,    # 是否保存最佳模型
        'save_final_model': True    # 是否保存最终模型
    }
}


def create_model(config):
    """
    根据配置创建模型
    
    参数:
        config: 模型配置字典
        
    返回:
        model: 创建的模型实例
    """
    # 获取模型类型
    model_type = config['model_type']
    # 获取模型参数
    model_params = {
        'hidden_dim': config['params']['hidden_dim'],
        'num_layers': config['params']['num_layers'],
        'dropout': config['params']['dropout'],
        'prediction_tasks': config['params']['prediction_tasks']
    }
    
    # 确保input_dim被正确设置
    if 'input_dim' in config:
        model_params['input_dim'] = config['input_dim']
    else:
        # 如果没有指定input_dim，则使用默认值
        # 通常需要在创建数据集后才能确定input_dim，这里暂时设置为10
        model_params['input_dim'] = 10
    
    # 根据模型类型添加特定参数
    if model_type == "GNN":
        # 为GNN模型添加特定参数
        model_params['model_type'] = config['type_specific_params']['GNN']['model_type']
        model = MoleculeGNN(**model_params)
    elif model_type == "GAT":
        # 为GAT模型添加特定参数
        model_params['num_heads'] = config['type_specific_params']['GAT']['num_heads']
        model = MoleculeGAT(**model_params)
    elif model_type == "GIN":
        model = MoleculeGIN(**model_params)
    elif model_type == "JKNet":
        # 为JKNet模型添加特定参数
        model_params['jk_mode'] = config['type_specific_params']['JKNet']['jk_mode']
        model = MoleculeJKNet(**model_params)
    elif model_type == "Transformer":
        # 为Transformer模型添加特定参数
        model_params['num_heads'] = config['type_specific_params']['Transformer']['num_heads']
        model = MoleculeGraphTransformer(**model_params)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
        
    return model


def load_data(config):
    """
    根据配置加载数据
    
    参数:
        config: 数据配置字典
        
    返回:
        train_loader, val_loader, test_loader: 数据加载器
    """
    # 创建数据集
    dataset = MoleculeDataset(
        root=config['data_dir'],
        filename=config['csv_file'],
        smiles_col=config['smiles_col'],
        label_col=config['label_col']
    )
    
    # 获取数据集划分
    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(
        indices, 
        test_size=config['test_size'] + config['val_size'], 
        random_state=config['random_state']
    )
    
    # 进一步划分验证集和测试集
    val_size_relative = config['val_size'] / (config['test_size'] + config['val_size'])
    val_indices, test_indices = train_test_split(
        temp_indices, 
        test_size=val_size_relative, 
        random_state=config['random_state']
    )
    
    # 创建子数据集（使用索引直接访问）
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # 返回输入维度
    input_dim = dataset[0].x.shape[1] if len(dataset) > 0 else 10
    
    return train_loader, val_loader, test_loader, input_dim


def pretrain_model():
    """
    执行模型预训练
    """
    print("开始模型预训练...")
    
    # 获取配置
    data_config = PRETRAIN_CONFIG['data_config']
    model_config = PRETRAIN_CONFIG['model_config']
    training_config = PRETRAIN_CONFIG['training_config']
    save_config = PRETRAIN_CONFIG['save_config']
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    # 输出当前处于的绝对路径和路径下的文件夹.
    print(os.getcwd(), os.listdir())
    train_loader, val_loader, test_loader, input_dim = load_data(data_config)
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 更新模型配置中的输入维度
    model_config['input_dim'] = input_dim
    
    # 创建模型
    print(f"创建 {model_config['model_type']} 模型...")
    model = create_model(model_config)
    model = model.to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数数量: {total_params}")
    
    # 定义损失函数和优化器
    criterion = UncertaintyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    
    # 创建TensorBoard写入器
    writer = create_tensorboard_writer()
    
    # 记录配置信息
    log_config_to_tensorboard(writer, data_config, 'Data')
    log_config_to_tensorboard(writer, model_config, 'Model')
    log_config_to_tensorboard(writer, training_config, 'Training')
    
    print("开始训练...")
    
    # 训练模型
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in tqdm(range(training_config['epochs']), position=0, leave=True):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # 前向传播
            output, uncertainty = model(data.x, data.edge_index, data.batch)
            loss = criterion(output, data.y, uncertainty)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 累计损失
            train_loss += loss.item() * data.num_graphs
            train_samples += data.num_graphs
            
        # 计算平均训练损失
        avg_train_loss = train_loss / train_samples
        train_losses.append(avg_train_loss)
        
        # 验证阶段
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
                
        # 计算平均验证损失
        avg_val_loss = val_loss / val_samples
        val_losses.append(avg_val_loss)
        
        # 记录TensorBoard日志
        log_losses_to_tensorboard(writer, avg_train_loss, avg_val_loss, epoch)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss and save_config['save_best_model']:
            best_val_loss = avg_val_loss
            best_model_path = get_model_save_path("pretrained_best_model.pth")
            save_model(
                model, 
                model_config['model_type'], 
                model_config, 
                best_model_path, 
                epoch, 
                avg_train_loss, 
                avg_val_loss
            )
            print(f"在第 {epoch+1} 轮保存了最佳模型，验证损失: {avg_val_loss:.4f}")
        
        # 每隔一定轮数保存检查点
        if (epoch + 1) % save_config['checkpoint_interval'] == 0:
            checkpoint_path = get_model_save_path(f"pretrained_checkpoint_epoch_{epoch+1}.pth")
            save_model(
                model,
                model_config['model_type'],
                model_config,
                checkpoint_path,
                epoch,
                avg_train_loss,
                avg_val_loss,
                optimizer.state_dict()
            )
    
    # 保存最终模型
    if save_config['save_final_model']:
        final_model_path = get_model_save_path("pretrained_final_model.pth")
        save_model(
            model,
            model_config['model_type'],
            model_config,
            final_model_path,
            training_config['epochs'] - 1,
            train_losses,
            val_losses
        )
        print(f"最终模型已保存到: {final_model_path}")
    
    # 关闭TensorBoard写入器
    close_tensorboard_writer(writer)
    
    print("预训练完成!")
    
    # 在测试集上评估
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
    print(f"测试集损失: {avg_test_loss:.4f}")
    
    return model, train_losses, val_losses


if __name__ == "__main__":
    # 运行预训练
    pretrain_model()