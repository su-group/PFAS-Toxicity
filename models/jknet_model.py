#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JKNet模型模块

该模块定义基于PyG的跳跃知识网络模型，适配当前分子图结构:
1. 使用Jumping Knowledge Network处理分子图
2. 融合多层信息增强表达能力
3. 保持与其他模型一致的接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, JumpingKnowledge


class MoleculeJKNet(nn.Module):
    """
    分子性质预测的跳跃知识网络模型
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 mode: str = 'cat',
                 dropout: float = 0.2,
                 prediction_tasks: int = 1):
        """
        初始化JKNet模型

        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            mode: 跳跃知识聚合模式 ('cat', 'max', 'lstm')
            dropout: Dropout率
            prediction_tasks: 预测任务数
        """
        super(MoleculeJKNet, self).__init__()
        self.num_layers = num_layers
        self.mode = mode
        self.dropout = dropout
        self.prediction_tasks = prediction_tasks

        # 构建GCN层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 第一层
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 中间层
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # 跳跃知识聚合器
        self.jump = JumpingKnowledge(mode=mode, channels=hidden_dim, num_layers=num_layers)

        # 输出层维度取决于聚合模式
        if mode == 'cat':
            output_dim = hidden_dim * num_layers
        else:
            output_dim = hidden_dim
            
        self.pool = global_mean_pool
        self.dropout_layer = nn.Dropout(dropout)
        
        # 输出层（需要考虑跳跃连接后的维度）
        self.predictor = nn.Linear(output_dim, prediction_tasks)
        
        # 不确定度预测
        self.uncertainty_predictor = nn.Linear(output_dim, prediction_tasks)

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
        # 存储每层的表示
        xs = []
        
        # GCN传播
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout_layer(x)
            xs.append(x)

        # 跳跃知识聚合
        x = self.jump(xs)

        # 全局池化
        x = self.pool(x, batch)

        # 预测和不确定度
        predictions = self.predictor(x)
        uncertainties = torch.sigmoid(self.uncertainty_predictor(x))

        return predictions, uncertainties


# 测试代码
if __name__ == "__main__":
    # 创建示例模型
    model = MoleculeJKNet(input_dim=11, hidden_dim=32, num_layers=3, mode='cat')
    print(f"模型结构: {model}")