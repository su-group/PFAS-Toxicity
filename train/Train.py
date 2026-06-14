#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练脚本

该脚本提供完整的模型训练功能:
1. 数据准备和预处理
2. 模型创建和配置
3. 模型训练流程
4. 模型保存和配置导出
"""

import torch
import numpy as np
import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from sklearn.preprocessing import StandardScaler
    
    # 导入自定义模块
    from dataset.molecule_dataset import MoleculeDataset
    from dataset.data_utils import prepare_data_from_files
    from models import MoleculeGNN, MoleculeGAT, MoleculeGIN, MoleculeGraphTransformer, MoleculeJKNet, UncertaintyLoss
    from train.trainer import MoleculeTrainer
    from config import (
        DATA_CONFIG, 
        MODEL_CONFIG, 
        TRAINING_CONFIG, 
        get_model_params, 
        get_model_save_path, 
        save_config_to_output,
        TASK_OUTPUT_DIR
    )
    from torch_geometric.loader import DataLoader  # 修复DataLoader导入警告
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保已安装所有依赖项")
    sys.exit(1)


def prepare_datasets(data_path, batch_size=32, normalize_labels=True):
    """
    根据输入路径准备数据集
    
    参数:
        data_path: 数据路径，可以是文件夹或CSV文件
        batch_size: 批处理大小
        normalize_labels: 是否对标签进行标准化
        
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        scaler: 标准化器（如果需要）
    """
    try:
        # 使用数据处理工具准备数据
        folder_path = prepare_data_from_files(data_path)
        print(f"处理数据文件夹: {folder_path}")
        
        # 创建数据集实例
        train_dataset = MoleculeDataset(
            root=folder_path, 
            filename="train_filled.csv",
            smiles_col=DATA_CONFIG['smiles_col'], 
            label_col=DATA_CONFIG['label_col']
        )
        val_dataset = MoleculeDataset(
            root=folder_path, 
            filename="eval.csv", 
            smiles_col=DATA_CONFIG['smiles_col'], 
            label_col=DATA_CONFIG['label_col']
        )
        test_dataset = MoleculeDataset(
            root=folder_path, 
            filename="test_filled.csv",
            smiles_col=DATA_CONFIG['smiles_col'], 
            label_col=DATA_CONFIG['label_col']
        )
        
        # 显示训练集统计信息
        print("\n训练集统计信息:")
        stats = train_dataset.get_property_stats()
        if stats:
            for key, value in stats.items():
                print(f"  {key}: {value:.4f}")
        
        # 如果需要标准化标签，则进行标准化
        scaler = None
        if normalize_labels:
            # 获取所有标签值
            all_labels = []
            for dataset in [train_dataset, val_dataset, test_dataset]:
                for data in dataset:
                    all_labels.append(data.y.item())
            
            # 创建标准化器并拟合
            scaler = StandardScaler()
            all_labels = np.array(all_labels).reshape(-1, 1)
            scaler.fit(all_labels)
            
            # 应用标准化到数据集
            for dataset in [train_dataset, val_dataset, test_dataset]:
                for i in range(len(dataset)):
                    dataset[i].y = torch.tensor(
                        scaler.transform(dataset[i].y.reshape(1, -1)).flatten(),
                        dtype=torch.float32
                    )
            
            print("标签已标准化")
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, scaler
    except Exception as e:
        print(f"数据准备失败: {e}")
        raise


def create_model(model_type, input_dim, model_params):
    """
    创建指定类型的模型
    
    参数:
        model_type: 模型类型 ('GNN', 'GAT', 'GIN', 'Transformer', 'JKNet')
        input_dim: 输入特征维度
        model_params: 模型参数字典
        
    返回:
        model: 创建的模型实例
    """
    model_constructors = {
        'GNN': MoleculeGNN,
        'GAT': MoleculeGAT,
        'GIN': MoleculeGIN,
        'Transformer': MoleculeGraphTransformer,
        'JKNet': MoleculeJKNet
    }
    
    if model_type not in model_constructors:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 过滤掉不适用于特定模型构造函数的参数
    filtered_params = {k: v for k, v in model_params.items() 
                      if k not in ['hidden_dim', 'num_layers', 'dropout', 'prediction_tasks']}
    
    model = model_constructors[model_type](
        input_dim=input_dim,
        hidden_dim=model_params.get('hidden_dim', 64),
        num_layers=model_params.get('num_layers', 3),
        dropout=model_params.get('dropout', 0.2),
        prediction_tasks=model_params.get('prediction_tasks', 1),
        **filtered_params
    )
    
    return model


def train_model():
    """
    训练模型主函数
    """
    print("分子图建模与性质预测系统 - 训练模式")
    print("=" * 50)
    
    # 保存配置到输出目录
    print(f"任务输出目录: {TASK_OUTPUT_DIR}")
    config_path = save_config_to_output()
    print(f"配置已保存到: {config_path}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备数据
    try:
        train_loader, val_loader, test_loader, scaler = prepare_datasets(
            DATA_CONFIG['data_dir'], 
            DATA_CONFIG['batch_size'],
            DATA_CONFIG['normalize_labels']
        )
        
        # 保存标准化器参数（如果存在）
        if scaler is not None:
            scaler_path = os.path.join(TASK_OUTPUT_DIR, "label_scaler.npy")
            np.save(scaler_path, {
                'mean': scaler.mean_,
                'scale': scaler.scale_
            })
            print(f"标签标准化参数已保存: {scaler_path}")
    except Exception as e:
        print(f"数据准备失败: {e}")
        return
    
    # 获取输入维度
    try:
        sample_data = next(iter(train_loader))
        input_dim = sample_data.x.shape[1]
        print(f"输入特征维度: {input_dim}")
    except Exception as e:
        print(f"无法获取输入维度: {e}")
        return
    
    # 创建模型
    try:
        model_type = MODEL_CONFIG['default_type']
        model_params = get_model_params(model_type)
        model = create_model(model_type, input_dim, model_params)
        model = model.to(device)
        print(f"创建模型: {model_type}")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    except Exception as e:
        print(f"模型创建失败: {e}")
        return
    
    # 创建损失函数和优化器
    criterion = UncertaintyLoss(alpha=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # 创建训练器
    trainer = MoleculeTrainer(model, criterion, optimizer, device)
    
    # 训练模型
    print("\n开始训练模型...")
    try:
        train_losses, val_losses = trainer.train(
            train_loader, 
            val_loader, 
            epochs=TRAINING_CONFIG['epochs']
        )
    except Exception as e:
        print(f"模型训练失败: {e}")
        return
    
    # 保存最终模型
    try:
        final_model_path = get_model_save_path("final_model.pth")
        torch.save(model.state_dict(), final_model_path, weights_only=True)
        print(f"最终模型已保存: {final_model_path}")
    except Exception as e:
        print(f"模型保存失败: {e}")
    
    print("\n模型训练完成!")


if __name__ == "__main__":
    train_model()