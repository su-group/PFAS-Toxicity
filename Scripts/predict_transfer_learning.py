#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迁移学习模型预测脚本

使用训练好的迁移学习模型对CSV数据集中的分子进行预测，
并将预测结果保存为CSV文件。
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import periodictable
from rdkit import Chem

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gin_model import MoleculeGIN


MODEL_CONFIG = {
    "input_dim": 120,
    "hidden_dim": 256,
    "num_layers": 8,
    "dropout": 0.2,
    "prediction_tasks": 1
}

CHECKPOINT_PATH = r"F:\GNN-pro\scripts\outputs\GIN_20260107-094723\GIN_TL_Start_cf2_510_task_20260528-184435\transfer_learned_best_model.pth"

DATA_CSV_PATH = r"F:\GNN-pro\data_process_class\step3_OECD_Class_Enhanced_v2_with_CF.csv"

OUTPUT_CSV_PATH = r"F:\GNN-pro\data_process_class\predict_oecd_cf2_GIN510_v3.csv"

MEAN_VAL = 2.9348
STD_VAL = 1.0857


class SimpleMoleculeGraphBuilder:
    """简化的分子图构建器"""

    def __init__(self, max_atoms=256, input_dim=120):
        self.max_atoms = max_atoms
        self.input_dim = input_dim
        self.atom_types = [el.symbol for el in periodictable.elements]
        self.atom_type_to_idx = {atom: idx for idx, atom in enumerate(self.atom_types)}

    def smiles_to_pyg_data(self, smiles: str) -> Data:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"无法解析SMILES: {smiles}")

            mol = Chem.AddHs(mol)
            num_atoms = mol.GetNumAtoms()

            atom_features = torch.zeros((self.max_atoms, self.input_dim))
            for i in range(min(num_atoms, mol.GetNumAtoms())):
                atom = mol.GetAtomWithIdx(i)
                symbol = atom.GetSymbol()
                if symbol in self.atom_type_to_idx:
                    atom_features[i, self.atom_type_to_idx[symbol]] = 1
                else:
                    atom_features[i, -1] = 1

            for i in range(num_atoms, self.max_atoms):
                atom_features[i, -1] = 1

            adjacency_matrix = torch.zeros((self.max_atoms, self.max_atoms))
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                if i < self.max_atoms and j < self.max_atoms:
                    bond_type = bond.GetBondTypeAsDouble()
                    adjacency_matrix[i, j] = bond_type
                    adjacency_matrix[j, i] = bond_type

            edge_index = adjacency_matrix.nonzero().t().contiguous()
            edge_attr = adjacency_matrix[edge_index[0], edge_index[1]]

            data = Data(
                x=atom_features.float(),
                edge_index=edge_index.long(),
                edge_attr=edge_attr.float()
            )

            return data
        except Exception as e:
            raise ValueError(f"处理SMILES {smiles} 时出错: {str(e)}")


def load_model_from_checkpoint(checkpoint_path):
    """从检查点加载模型"""
    print(f"正在加载模型检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model = MoleculeGIN(**MODEL_CONFIG)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    print(f"模型已加载成功!")
    print(f"检查点信息: epoch={checkpoint.get('epoch', 'N/A')}")
    return model


def create_dataset_from_csv(csv_path, smiles_col='SMILES'):
    """从CSV文件创建数据集"""
    print(f"正在读取CSV文件: {csv_path}")

    df = pd.read_csv(csv_path)
    smiles_list = df[smiles_col].tolist()

    print(f"共读取 {len(smiles_list)} 个分子")

    builder = SimpleMoleculeGraphBuilder()
    data_list = []
    failed_indices = []

    print("正在将SMILES转换为图结构...")
    for idx, smiles in enumerate(tqdm(smiles_list)):
        try:
            data = builder.smiles_to_pyg_data(smiles)
            data_list.append(data)
        except Exception as e:
            failed_indices.append(idx)
            print(f"处理分子 {idx} 失败: {smiles[:50]}... 错误: {str(e)}")

    print(f"转换完成: 成功 {len(data_list)} 个, 失败 {len(failed_indices)} 个")

    return data_list, df, smiles_list


def predict(model, data_list, device='cuda', batch_size=32):
    """对数据集进行预测"""
    model = model.to(device)
    model.eval()

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    predictions = []
    uncertainties = []

    print("正在进行预测...")
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device)
            pred, unc = model(data.x, data.edge_index, data.batch)

            predictions.extend(pred.cpu().numpy().flatten())
            uncertainties.extend(unc.cpu().numpy().flatten())

    return np.array(predictions), np.array(uncertainties)


def save_predictions(original_df, predictions, uncertainties, output_path):
    """保存预测结果"""
    preds = np.array(predictions, dtype=np.float32) * STD_VAL + MEAN_VAL

    results_df = original_df.copy()
    results_df['Predicted_NeglogLD50'] = preds
    results_df['Uncertainty'] = uncertainties

    results_df.to_csv(output_path, index=False)
    print(f"预测结果已保存到: {output_path}")

    print("\n反标准化后的预测结果统计:")
    print(f"  预测值范围: [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"  预测值均值: {preds.mean():.4f}")
    print(f"  预测值标准差: {preds.std():.4f}")
    print(f"  不确定度范围: [{uncertainties.min():.4f}, {uncertainties.max():.4f}]")
    print(f"  不确定度均值: {uncertainties.mean():.4f}")


def main():
    print("=" * 60)
    print("迁移学习模型预测")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = load_model_from_checkpoint(CHECKPOINT_PATH)

    data_list, original_df, smiles_list = create_dataset_from_csv(DATA_CSV_PATH)

    if len(data_list) == 0:
        print("错误: 没有成功处理任何分子!")
        return

    predictions, uncertainties = predict(model, data_list, device)

    save_predictions(original_df, predictions, uncertainties, OUTPUT_CSV_PATH)

    print("\n预测完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
