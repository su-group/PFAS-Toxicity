#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型测试模块

该模块实现模型测试流程，包括:
1. 加载训练好的模型
2. 在测试集上评估模型性能
3. 生成评估指标和结果分析
"""

import torch
import os
import sys
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from ..dataset import graph_builder
from ..dataset import molecule_dataset
from ..dataset import data_utils
from ..models import gnn_model
from ..eval import evaluator
from torch_geometric.loader import DataLoader


def load_model(model_path, device, num_atom_types=11):
    """
    加载训练好的模型

    参数:
        model_path: 模型文件路径
        device: 计算设备
        num_atom_types: 原子类型数量

    返回:
        model: 加载的模型
    """
    # 创建模型
    model = gnn_model.MoleculeGNN(
        input_dim=num_atom_types,
        hidden_dim=64,
        num_layers=3,
        model_type="GCN",
        dropout=0.2,
        prediction_tasks=1
    ).to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def test_model(data_path, model_path, device='cpu'):
    """
    测试模型性能

    参数:
        data_path: 数据路径
        model_path: 模型文件路径
        device: 计算设备
    """
    print("=== 模型测试 ===")
    
    # 准备数据
    print("\n1. 准备数据...")
    folder_path = data_utils.prepare_data_from_files(data_path)
    test_dataset = molecule_dataset.MoleculeDataset(root=folder_path, filename="test_filled.csv")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 获取原子类型数量
    builder = graph_builder.MoleculeGraphBuilder()
    num_atom_types = len(builder.atom_types) + 1
    print(f"原子类型数量: {num_atom_types}")
    
    # 加载模型
    print("\n2. 加载模型...")
    model = load_model(model_path, device, num_atom_types)
    print(f"模型已加载: {model_path}")
    
    # 创建评估器
    print("\n3. 模型评估...")
    evaluator_instance = evaluator.MoleculeEvaluator(model, device)
    
    # 在测试集上评估
    metrics, y_true, y_pred, uncertainties = evaluator_instance.evaluate(test_loader)
    
    print("测试集评估结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return metrics, y_true, y_pred, uncertainties


# 测试代码
if __name__ == "__main__":
    print("模型测试模块已创建")
    print("请使用完整测试脚本进行模型测试")