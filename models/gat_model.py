#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAT模型模块

该模块定义基于PyG的图注意力网络模型，适配当前分子图结构:
1. 使用Graph Attention Network处理分子图
2. 支持多头注意力机制
3. 保持与其他模型一致的接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class MoleculeGAT(nn.Module):
    """
    分子性质预测的图注意力网络模型
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 prediction_tasks: int = 1):
        """
        初始化GAT模型

        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            num_heads: 注意力头数
            dropout: Dropout率
            prediction_tasks: 预测任务数
        """
        super(MoleculeGAT, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.prediction_tasks = prediction_tasks

        # 构建GAT层
        self.gat_layers = nn.ModuleList()
        
        # 第一层 (input_dim -> hidden_dim)
        self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        
        # 中间层 (hidden_dim * num_heads -> hidden_dim)
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))

        # 输出层 (hidden_dim * num_heads -> hidden_dim, 1 head)
        self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout))

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
        # GAT传播
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index)
            # 最后一层不需要激活函数
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
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
    model = MoleculeGAT(input_dim=11, hidden_dim=32, num_layers=2, num_heads=4)
    print(f"模型结构: {model}")