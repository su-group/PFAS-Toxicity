#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子图构建模块

该模块负责将SMILES字符串转换为图结构数据，包括:
1. 解析SMILES字符串
2. 生成邻接矩阵（键值对键作为bond order）
3. 生成特征矩阵（原子类型one-hot编码）
4. 验证矩阵对称性及维度规范
"""
import periodictable
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import periodictable

# 导入配置管理器
from config import get_data_config

class MoleculeGraphBuilder:
    """
    分子图构建器，用于将SMILES字符串转换为图结构数据
    """

    def __init__(self, max_atoms: int = None):
        """
        初始化分子图构建器

        参数:
            max_atoms: 最大原子数，用于统一图大小，如果为None则从配置中获取
        """
        data_config = get_data_config()
        self.max_atoms = max_atoms if max_atoms is not None else data_config.get("max_atoms", 256)
        # 原子类型列表，用于one-hot编码，用元素周期表中
        self.atom_types = [el.symbol for el in periodictable.elements]
        # self.atom_types = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        self.atom_type_to_idx = {atom: idx for idx, atom in enumerate(self.atom_types)}

    def smiles_to_graph(self, smiles: str) -> tuple:
        """
        将SMILES字符串转换为图结构（邻接矩阵和特征矩阵）

        参数:
            smiles: 分子的SMILES表示

        返回:
            atom_features: 原子特征矩阵 (N, F)
            adjacency_matrix: 邻接矩阵 (N, N)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"无法解析SMILES: {smiles}")

            # 添加氢原子
            mol = Chem.AddHs(mol)

            # 获取原子数
            num_atoms = mol.GetNumAtoms()
            if num_atoms > self.max_atoms:
                print(f"警告: 分子原子数({num_atoms})超过最大限制({self.max_atoms})")
                num_atoms = self.max_atoms

            # 构建特征矩阵 (one-hot编码)
            atom_features = torch.zeros((self.max_atoms, len(self.atom_types) + 1))  # +1 for padding

            for i in range(min(num_atoms, mol.GetNumAtoms())):
                atom = mol.GetAtomWithIdx(i)
                symbol = atom.GetSymbol()
                if symbol in self.atom_type_to_idx:
                    atom_features[i, self.atom_type_to_idx[symbol]] = 1
                else:
                    # 未知原子类型标记为最后一类
                    atom_features[i, -1] = 1

            # 填充位置标记为最后一类
            for i in range(num_atoms, self.max_atoms):
                atom_features[i, -1] = 1

            # 构建邻接矩阵 (键值作为边权重)
            adjacency_matrix = torch.zeros((self.max_atoms, self.max_atoms))

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                if i < self.max_atoms and j < self.max_atoms:
                    bond_type = bond.GetBondTypeAsDouble()
                    adjacency_matrix[i, j] = bond_type
                    adjacency_matrix[j, i] = bond_type

            return atom_features, adjacency_matrix

        except Exception as e:
            raise ValueError(f"处理SMILES {smiles} 时出错: {str(e)}")

    def validate_graph(self, atom_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> bool:
        """
        验证图结构的正确性

        参数:
            atom_features: 原子特征矩阵
            adjacency_matrix: 邻接矩阵

        返回:
            bool: 是否有效
        """
        # 检查矩阵维度
        if atom_features.shape != (self.max_atoms, len(self.atom_types) + 1):
            return False

        if adjacency_matrix.shape != (self.max_atoms, self.max_atoms):
            return False

        # 检查邻接矩阵对称性
        if not torch.allclose(adjacency_matrix, adjacency_matrix.t()):
            return False

        return True

    # def smiles_to_pyg_data(self, smiles: str) -> Data:
    #     """
    #     将SMILES转换为PyG Data对象
    #
    #     参数:
    #         smiles: 分子的SMILES表示
    #
    #     返回:
    #         Data: PyG数据对象
    #     """
    #     # 构建图
    #     atom_features, adjacency_matrix = self.smiles_to_graph(smiles)
    #
    #     # 验证图结构
    #     if not self.validate_graph(atom_features, adjacency_matrix):
    #         raise ValueError("图结构验证失败")
    #
    #     # 转换为边索引格式
    #     edge_index = adjacency_matrix.nonzero().t().contiguous()
    #     edge_attr = adjacency_matrix[edge_index[0], edge_index[1]]
    #
    #     # 创建PyG Data对象
    #     data = Data(
    #         x=atom_features.float(),
    #         edge_index=edge_index.long(),
    #         edge_attr=edge_attr.float()
    #     )
    #
    #     return data
    def smiles_to_pyg_data(self, smiles: str) -> Data:
        """
        将SMILES转换为PyG Data对象
        """
        # 构建图 (atom_features 形状: [max_atoms, F])
        atom_features, adjacency_matrix = self.smiles_to_graph(smiles)

        if not self.validate_graph(atom_features, adjacency_matrix):
            raise ValueError("图结构验证失败")

        # === 关键修改：确定真实原子数量 ===
        # 假设填充的原子其最后一维特征为1，真实原子为0
        # 检查每一行，如果最后一维是1，则认为是填充原子
        padding_mask = atom_features[:, -1] == 1
        # 找到第一个填充原子的位置，这就是真实原子的数量
        padding_indices = padding_mask.nonzero(as_tuple=True)[0]
        if len(padding_indices) > 0:
            num_real_atoms = padding_indices[0].item()
        else:
            num_real_atoms = self.max_atoms
        # === 关键修改结束 ===

        # 转换为边索引格式
        edge_index = adjacency_matrix.nonzero().t().contiguous()
        edge_attr = adjacency_matrix[edge_index[0], edge_index[1]]

        # 创建PyG Data对象，并添加新属性
        data = Data(
            x=atom_features.float(),
            edge_index=edge_index.long(),
            edge_attr=edge_attr.float(),
            num_real_atoms=num_real_atoms  # 👈 添加此行
        )

        return data


def create_sample_dataset(smiles_list: list, properties: list) -> list:
    """
    创建示例数据集

    参数:
        smiles_list: SMILES字符串列表
        properties: 对应的分子性质列表

    返回:
        data_list: PyG Data对象列表
    """
    builder = MoleculeGraphBuilder()
    data_list = []

    for smiles, prop in zip(smiles_list, properties):
        try:
            data = builder.smiles_to_pyg_data(smiles)
            data.y = torch.tensor([prop], dtype=torch.float)
            data_list.append(data)
        except Exception as e:
            print(f"处理分子 {smiles} 时出错: {e}")

    return data_list


# 测试代码
if __name__ == "__main__":
    # 测试分子图构建
    print("测试分子图构建...")
    builder = MoleculeGraphBuilder(max_atoms=10)

    # 测试简单分子
    test_smiles = "CCO"  # 乙醇
    features, adj = builder.smiles_to_graph(test_smiles)
    print(f"SMILES: {test_smiles}")
    print(f"特征矩阵形状: {features.shape}")
    print(f"邻接矩阵形状: {adj.shape}")
    print(f"图结构验证: {builder.validate_graph(features, adj)}")

    # 测试PyG Data对象转换
    data = builder.smiles_to_pyg_data(test_smiles)
    print(f"PyG Data对象: {data}")
    print(f"x形状: {data.x.shape}")
    print(f"edge_index形状: {data.edge_index.shape}")
    print(f"edge_attr形状: {data.edge_attr.shape}")