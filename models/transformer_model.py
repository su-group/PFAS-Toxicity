#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph Transformer模型模块

该模块定义基于Transformer的图神经网络模型，适配当前分子图结构:
1. 使用自注意力机制处理分子图
2. 支持全局信息交互
3. 保持与其他模型一致的接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv


class MoleculeGraphTransformer(nn.Module):
    """
    分子性质预测的图Transformer模型
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 prediction_tasks: int = 1):
        """
        初始化Graph Transformer模型

        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            num_heads: 注意力头数
            dropout: Dropout率
            prediction_tasks: 预测任务数
        """
        super(MoleculeGraphTransformer, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.prediction_tasks = prediction_tasks
        self.hidden_dim = hidden_dim

        # 节点特征嵌入
        self.node_embedding = nn.Linear(input_dim, hidden_dim)

        # 多头自注意力层
        self.self_attn_layers = nn.ModuleList()
        self.gcn_layers = nn.ModuleList()
        self.layer_norms1 = nn.ModuleList()
        self.layer_norms2 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        for _ in range(num_layers):
            # 多头自注意力
            self.self_attn_layers.append(nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ))

            # GCN层用于捕捉局部结构信息
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

            # Layer normalization
            self.layer_norms1.append(nn.LayerNorm(hidden_dim))
            self.layer_norms2.append(nn.LayerNorm(hidden_dim))

            # 前馈网络
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            ))

        # 输出层
        self.pool = global_mean_pool
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
        # 节点嵌入
        h = self.node_embedding(x)

        # 获取每个图的节点数量
        batch_size = batch.max().item() + 1
        batch_counts = torch.bincount(batch)
        max_nodes = batch_counts.max().item()

        # 为每个图构建注意力掩码
        masks = []
        for i in range(batch_size):
            node_count = batch_counts[i]
            mask = torch.zeros(max_nodes, dtype=torch.bool, device=x.device)
            mask[:node_count] = True
            masks.append(mask)

        # 将节点特征重塑为批次格式用于注意力计算
        # 这里我们使用一个简化的方法，将所有节点视为一个序列
        # 实际应用中可能需要更复杂的批处理方法

        # 多层Transformer
        for i in range(self.num_layers):
            # 1. 图卷积获取局部信息
            local_h = self.gcn_layers[i](h, edge_index)

            # 2. 自注意力机制获取全局信息
            # 注意: 这里简化处理，实际应用中需要更好的批处理方法
            attn_out, _ = self.self_attn_layers[i](local_h, local_h, local_h)

            # 3. 残差连接和LayerNorm
            h = self.layer_norms1[i](h + attn_out)

            # 4. 前馈网络
            ffn_out = self.ffn_layers[i](h)

            # 5. 残差连接和LayerNorm
            h = self.layer_norms2[i](h + ffn_out)

        # 全局池化
        x = self.pool(h, batch)

        # 预测和不确定度
        predictions = self.predictor(x)
        uncertainties = torch.sigmoid(self.uncertainty_predictor(x))

        return predictions, uncertainties


# 测试代码
if __name__ == "__main__":
    # 创建示例模型
    model = MoleculeGraphTransformer(input_dim=11, hidden_dim=32, num_layers=2, num_heads=4)
    print(f"模型结构: {model}")