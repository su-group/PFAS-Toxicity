#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GIN模型模块

该模块定义基于PyG的图同构网络模型，适配当前分子图结构:
1. 使用Graph Isomorphism Network处理分子图
2. 支持多层MLP进行节点更新
3. 保持与其他模型一致的接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool


class MoleculeGIN(nn.Module):
    """
    分子性质预测的图同构网络模型
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 prediction_tasks: int = 1):
        """
        初始化GIN模型

        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            dropout: Dropout率
            prediction_tasks: 预测任务数
        """
        super(MoleculeGIN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.prediction_tasks = prediction_tasks

        # 构建GIN层
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # 第一层
        mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.gin_layers.append(GINConv(mlp1, train_eps=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # 中间层
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            self.gin_layers.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

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
        # GIN传播
        for layer, batch_norm in zip(self.gin_layers, self.batch_norms):
            x = layer(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout_layer(x)

        # 全局池化
        x = self.pool(x, batch)

        # 预测和不确定度
        predictions = self.predictor(x)
        uncertainties = torch.sigmoid(self.uncertainty_predictor(x))

        return predictions, uncertainties


# 测试代码
if __name__ == "__main__":
    # 创建示例模型
    model = MoleculeGIN(input_dim=11, hidden_dim=32, num_layers=2)
    print(f"模型结构: {model}")