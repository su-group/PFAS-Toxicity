#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理变换模块

该模块定义了用于分子图数据预处理的变换类，可以与PyTorch Geometric的DataLoader配合使用。
"""

import torch
from torch_geometric.data import Data
import numpy as np


class NormalizeTarget:
    """
    标准化目标值（标签）的变换类
    """

    def __init__(self, mean=None, std=None, train_dataset=None):
        """
        初始化标准化变换
        
        参数:
            mean: 均值，如果为None且提供了train_dataset，则从训练数据计算
            std: 标准差，如果为None且提供了train_dataset，则从训练数据计算
            train_dataset: 训练数据集，用于计算均值和标准差
        """
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
        elif train_dataset is not None:
            self._compute_stats(train_dataset)
        else:
            raise ValueError("必须提供均值和标准差，或者提供训练数据集来计算统计信息")

    def _compute_stats(self, train_dataset):
        """
        从训练数据集中计算均值和标准差
        
        参数:
            train_dataset: 训练数据集
        """
        targets = []
        for data in train_dataset:
            targets.append(data.y.item())
        
        self.mean = np.mean(targets)
        self.std = np.std(targets)
        
        print(f"计算得到的目标值统计信息: 均值={self.mean:.4f}, 标准差={self.std:.4f}")

    def __call__(self, data):
        """
        应用标准化变换
        
        参数:
            data: PyG Data对象
            
        返回:
            标准化后的Data对象
        """
        if self.std > 0:
            data.y = (data.y - self.mean) / self.std
        return data

    def inverse_transform(self, y):
        """
        反向变换，将标准化后的值还原为原始值
        
        参数:
            y: 标准化后的值
            
        返回:
            原始值
        """
        return y * self.std + self.mean


class AddSelfLoops:
    """
    为图数据添加自环的变换类
    """

    def __init__(self, attr_value=1.0):
        """
        初始化添加自环变换
        
        参数:
            attr_value: 自环边的属性值
        """
        self.attr_value = attr_value

    def __call__(self, data):
        """
        应用添加自环变换
        
        参数:
            data: PyG Data对象
            
        返回:
            添加自环后的Data对象
        """
        # 获取节点数
        num_nodes = data.x.size(0)
        
        # 创建自环边索引
        self_loop_index = torch.arange(0, num_nodes, dtype=torch.long)
        self_loop_edge_index = torch.stack([self_loop_index, self_loop_index], dim=0)
        
        # 添加自环到现有边索引
        if data.edge_index is not None:
            edge_index = torch.cat([data.edge_index, self_loop_edge_index], dim=1)
        else:
            edge_index = self_loop_edge_index
            
        # 处理边属性
        if data.edge_attr is not None and hasattr(data, 'edge_attr'):
            # 创建自环边属性
            self_loop_attr = torch.full((num_nodes, data.edge_attr.size(1)), 
                                      self.attr_value, dtype=data.edge_attr.dtype)
            edge_attr = torch.cat([data.edge_attr, self_loop_attr], dim=0)
        else:
            edge_attr = None
            
        # 更新Data对象
        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
            
        return data


class VirtualPadding:
    """
    为图数据添加虚拟节点填充以统一图大小的变换类
    """

    def __init__(self, max_nodes):
        """
        初始化虚拟填充变换
        
        参数:
            max_nodes: 最大节点数
        """
        self.max_nodes = max_nodes

    def __call__(self, data):
        """
        应用虚拟填充变换
        
        参数:
            data: PyG Data对象
            
        返回:
            填充后的Data对象
        """
        num_nodes = data.x.size(0)
        
        if num_nodes >= self.max_nodes:
            # 如果节点数超过最大值，截断
            data.x = data.x[:self.max_nodes]
            # 需要相应地过滤边索引和边属性
            mask = (data.edge_index[0] < self.max_nodes) & (data.edge_index[1] < self.max_nodes)
            data.edge_index = data.edge_index[:, mask]
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                data.edge_attr = data.edge_attr[mask]
        else:
            # 添加虚拟节点
            padding_size = self.max_nodes - num_nodes
            feature_dim = data.x.size(1)
            
            # 创建填充特征（零向量）
            padding_features = torch.zeros(padding_size, feature_dim, dtype=data.x.dtype)
            data.x = torch.cat([data.x, padding_features], dim=0)
            
            # 调整边索引（虚拟节点不参与连接）
            # 不需要特别处理，因为它们默认没有连接
            
        return data


# 测试代码
if __name__ == "__main__":
    print("数据预处理变换模块已创建")
    print("支持以下变换:")
    print("1. NormalizeTarget: 标准化目标值")
    print("2. AddSelfLoops: 添加自环")
    print("3. VirtualPadding: 虚拟节点填充")