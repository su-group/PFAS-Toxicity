#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义分子数据集类

该模块定义了用于处理分子数据的自定义数据集类，支持从CSV文件加载数据
并将其转换为PyTorch Geometric格式。
"""

import os
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data

# 添加PyG相关的安全全局变量
try:
    from torch_geometric.data import DataEdgeAttr, DataFaceAttr
    torch.serialization.add_safe_globals([DataEdgeAttr, DataFaceAttr])
except ImportError:
    pass

import os
import numpy as np
import shutil


class MoleculeDataset(InMemoryDataset):
    """
    自定义分子数据集类
    
    该类继承自PyTorch Geometric的InMemoryDataset，用于处理分子数据。
    支持从CSV文件加载SMILES和对应的分子性质数据。
    """

    def __init__(self, root, filename=None, transform=None, pre_transform=None, pre_filter=None,
                 smiles_col=None, label_col=None, normalize_labels=None):
        """
        初始化分子数据集

        参数:
            root: 数据集根目录
            filename: CSV文件名
            transform: 数据变换函数
            pre_transform: 预处理变换函数
            pre_filter: 预过滤函数
            smiles_col: SMILES列名
            label_col: 标签列名
            normalize_labels: 是否标准化标签
        """
        # 从配置中获取默认参数
        from config import get_data_config
        data_config = get_data_config()

        self.filename = filename if filename is not None else "Labeled-Gap-IsoelectronicSubstitution.csv"
        self.smiles_col = smiles_col if smiles_col is not None else data_config.get("smiles_col", "SMILES")
        self.label_col = label_col if label_col is not None else data_config.get("label_col", "Gap(eV)")
        self.normalize_labels = normalize_labels if normalize_labels is not None else data_config.get("normalize_labels", False)
        print("参数:",self.normalize_labels)

        # 延迟导入以避免循环导入
        from .graph_builder import MoleculeGraphBuilder
        self.builder = MoleculeGraphBuilder()
        self.mean = 0
        self.std = 1
        super(MoleculeDataset, self).__init__(root, transform, pre_transform, pre_filter)
        # 只有在处理过的文件存在时才加载
        if os.path.exists(self.processed_paths[0]):
            # 修复torch.load的weights_only参数问题
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """
        返回原始文件名列表
        """
        return [self.filename]

    @property
    def processed_file_names(self):
        """
        返回处理后的文件名列表
        """
        return ['data.pt']

    def download(self):
        """
        下载数据集文件
        对于本地文件，此方法为空实现
        """
        pass

    def process(self):
        """
        处理原始数据并保存为PyTorch Geometric格式
        """
        # 读取CSV文件
        csv_path = os.path.join(self.root, self.filename)
        df = pd.read_csv(csv_path)


        # 提取SMILES和性质数据
        smiles_list = df[self.smiles_col].tolist()
        properties = df[self.label_col].tolist()

        # 转换为PyG Data对象
        data_list = []
        successful_count = 0
        failed_count = 0

        for smiles, prop in zip(smiles_list, properties):
            try:
                # 将SMILES转换为图结构
                data = self.builder.smiles_to_pyg_data(smiles)
                # 添加标签
                data.y = torch.tensor([prop], dtype=torch.float)
                data_list.append(data)
                successful_count += 1
            except Exception as e:
                print(f"处理分子 {smiles} 时出错: {e}")
                failed_count += 1
                continue

        print(f"数据处理完成: 成功 {successful_count} 个, 失败 {failed_count} 个")

        # 标签标准化
        if self.normalize_labels:
            # 计算均值和标准差
            labels = torch.tensor([data.y.item() for data in data_list])
            self.mean = labels.mean()
            self.std = labels.std()

            # 标准化标签
            for data in data_list:
                data.y = (data.y - self.mean) / (self.std + 1e-8)  # 添加小量避免除以0

            # 保存标准化参数
            norm_params = {'mean': self.mean, 'std': self.std}
            os.makedirs(self.processed_dir, exist_ok=True)
            np.save(os.path.join(self.processed_dir, 'norm_params.npy'), norm_params)
            print(f"标签已标准化，均值: {self.mean:.4f}, 标准差: {self.std:.4f}")

        # 应用预处理变换
        if self.pre_filter is not None:
            original_count = len(data_list)
            data_list = [data for data in data_list if self.pre_filter(data)]
            print(f"预过滤后剩余 {len(data_list)} 个数据 (原 {original_count} 个)")

        if self.pre_transform is not None:
            original_count = len(data_list)
            data_list = [self.pre_transform(data) for data in data_list]
            print(f"预变换后剩余 {len(data_list)} 个数据 (原 {original_count} 个)")

        # 保存处理后的数据
        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)

        torch.save((data, slices), self.processed_paths[0])
        print(f"数据已保存到 {self.processed_paths[0]}")

    def get_property_stats(self):
        """
        获取数据集的性质统计信息
        
        返回:
            stats: 性质统计信息字典
        """
        if self.data.y is not None:
            y_values = self.data.y.numpy().flatten()  # 确保是一维数组
            return {
                'count': len(y_values),
                'mean': float(y_values.mean()),
                'std': float(y_values.std()),
                'min': float(y_values.min()),
                'max': float(y_values.max())
            }
        return None

    def enable_normalization(self):
        """启用标签标准化"""
        self.normalize_labels = True
        
    def disable_normalization(self):
        """禁用标签标准化"""
        self.normalize_labels = False
        
    def load_normalization_params(self):
        """加载标准化参数"""
        norm_path = os.path.join(self.processed_dir, 'norm_params.npy')
        if os.path.exists(norm_path):
            norm_params = np.load(norm_path, allow_pickle=True).item()
            self.mean = norm_params['mean']
            self.std = norm_params['std']
            print(f"已加载标准化参数: 均值={self.mean:.4f}, 标准差={self.std:.4f}")
        else:
            print("未找到标准化参数文件")

    def inverse_normalize_labels(self, y):
        """
        反向标准化标签值（兼容tensor和numpy）

        参数:
            y: 标准化后的标签值 (tensor 或 numpy array)

        返回:
            原始标签值 (与输入类型相同)
        """
        if not self.normalize_labels or self.std <= 0:
            return y

        # 确定输入类型
        is_tensor = torch.is_tensor(y)

        # 如果输入是张量，确保标准化参数也是张量
        if is_tensor:
            if not torch.is_tensor(self.std):
                # 将标准化参数转换为张量
                std = torch.tensor(self.std, dtype=y.dtype, device=y.device)
                mean = torch.tensor(self.mean, dtype=y.dtype, device=y.device)
            else:
                std = self.std
                mean = self.mean
            return y * std + mean
        else:
            # 输入是numpy数组，确保标准化参数也是numpy数组
            if torch.is_tensor(self.std):
                std = self.std.cpu().numpy()
                mean = self.mean.cpu().numpy()
            else:
                std = self.std
                mean = self.mean
            return y * std + mean


    def __repr__(self):
        """
        返回数据集的字符串表示
        """
        return f'MoleculeDataset({self.filename}, {len(self)})'


# 测试代码
if __name__ == "__main__":
    # 创建数据集实例
    dataset = MoleculeDataset(root="../data", filename="Labeled-Gap-IsoelectronicSubstitution.csv")
    print(f"数据集大小: {len(dataset)}")
    print(f"第一个数据样本: {dataset[0]}")
    
    # 显示数据集统计信息
    stats = dataset.get_property_stats()
    if stats:
        print("数据集统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")