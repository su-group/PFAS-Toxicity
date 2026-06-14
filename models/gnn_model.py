#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN模型模块

该模块定义基于PyG的图神经网络模型，包括:
1. 初始化GNN模型架构（包含GCN/GraphSAGE层）
2. 设计损失函数（包含性质预测损失和不确定度损失）
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool


class MoleculeGNN(nn.Module):
    """
    分子性质预测的图神经网络模型
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 model_type: str = "GCN",
                 dropout: float = 0.2,
                 prediction_tasks: int = 1):
        """
        初始化GNN模型

        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            model_type: 模型类型 ("GCN" 或 "SAGE")
            dropout: Dropout率
            prediction_tasks: 预测任务数
        """
        super(MoleculeGNN, self).__init__()
        self.model_type = model_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.prediction_tasks = prediction_tasks

        # 选择GNN层类型
        if model_type == "GCN":
            gnn_layer = GCNConv
        elif model_type == "SAGE":
            gnn_layer = SAGEConv
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 构建GNN层
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(gnn_layer(input_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.gnn_layers.append(gnn_layer(hidden_dim, hidden_dim))

        # 输出层
        self.pool = global_mean_pool
        self.dropout_layer = nn.Dropout(dropout)
        self.predictor = nn.Linear(hidden_dim, prediction_tasks)

        # 不确定度预测
        self.uncertainty_predictor = nn.Linear(hidden_dim, prediction_tasks)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> tuple:
        """
        前向传播

        参数:
            x: 节点特征
            edge_index: 边索引
            batch: 批次信息

        返回:
            predictions: 预测值
            uncertainties: 不确定度
        """
        # GNN传播
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            x = torch.relu(x)
            x = self.dropout_layer(x)

        # 全局池化
        x = self.pool(x, batch)

        # 预测和不确定度
        predictions = self.predictor(x)
        uncertainties = torch.sigmoid(self.uncertainty_predictor(x))

        return predictions, uncertainties


class UncertaintyLoss(nn.Module):
    """
    不确定度损失函数
    结合了预测损失和不确定度损失
    """

    def __init__(self, alpha: float = 0.1):
        """
        初始化不确定度损失函数

        参数:
            alpha: 不确定度损失的权重
        """
        super(UncertaintyLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, uncertainties: torch.Tensor) -> torch.Tensor:
        """
        计算损失函数

        参数:
            predictions: 预测值 [batch_size, 1]
            targets: 真实值 [batch_size] 或 [batch_size, 1]
            uncertainties: 不确定度 [batch_size, 1]

        返回:
            loss: 总损失
        """
        # 确保targets和predictions具有相同的维度
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        
        # 预测损失
        prediction_loss = self.mse_loss(predictions, targets)
        
        # 不确定度损失（与预测误差相关）
        uncertainty_loss = torch.mean(torch.abs(uncertainties - torch.abs(predictions - targets)))
        
        # 总损失
        total_loss = prediction_loss + self.alpha * uncertainty_loss
        
        return total_loss


# 测试代码
if __name__ == "__main__":
    # 创建示例模型
    model = MoleculeGNN(input_dim=11, hidden_dim=32, num_layers=2, model_type="GCN")
    print(f"模型结构: {model}")
    
    # 创建损失函数
    loss_fn = UncertaintyLoss(alpha=0.1)
    print(f"损失函数: {loss_fn}")