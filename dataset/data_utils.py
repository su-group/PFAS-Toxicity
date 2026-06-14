#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理工具模块

该模块提供数据处理相关的工具函数:
1. 支持不同的数据输入方式
2. 自动分割数据集
3. 数据路径管理
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

def prepare_data_from_files(data_path):
    """
    根据输入路径准备数据，支持文件夹和单个CSV文件输入
    
    参数:
        data_path: 数据路径，可以是文件夹或CSV文件
        
    返回:
        folder_path: 包含数据集的文件夹路径
    """
    # 检查路径是文件还是文件夹
    if os.path.isfile(data_path) and data_path.endswith('.csv'):
        # 处理单个CSV文件
        return prepare_data_from_single_csv(data_path)
    elif os.path.isdir(data_path):
        # 处理文件夹
        return prepare_data_from_folder(data_path)
    else:
        raise ValueError(f"不支持的数据路径格式: {data_path}")


def prepare_data_from_single_csv(csv_path):
    """
    从单个CSV文件准备数据，自动分割为训练/验证/测试集
    
    参数:
        csv_path: CSV文件路径
        
    返回:
        folder_path: 包含分割后数据集的文件夹路径
    """
    print(f"处理单个CSV文件: {csv_path}")
    
    # 创建与CSV文件同名的文件夹
    folder_name = os.path.splitext(os.path.basename(csv_path))[0]
    folder_path = os.path.join(os.path.dirname(csv_path), folder_name)
    
    # 创建文件夹（如果不存在）
    os.makedirs(folder_path, exist_ok=True)
    
    # 移动原CSV文件到新文件夹
    new_csv_path = os.path.join(folder_path, os.path.basename(csv_path))
    if not os.path.exists(new_csv_path):
        shutil.move(csv_path, new_csv_path)
        print(f"原CSV文件已移动到: {new_csv_path}")
    
    # 读取数据
    df = pd.read_csv(new_csv_path)
    
    # 分割数据
    train_df, temp_df = train_test_split(df, test_size=0.8, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # 保存分割后的数据
    train_path = os.path.join(folder_path, "train_filled.csv")
    val_path = os.path.join(folder_path, "eval.csv")  # 按要求命名为eval.csv
    test_path = os.path.join(folder_path, "test_filled.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"数据已分割并保存到文件夹: {folder_path}")
    print(f"  训练集: {len(train_df)} 样本")
    print(f"  验证集: {len(val_df)} 样本")
    print(f"  测试集: {len(test_df)} 样本")
    
    return folder_path


def prepare_data_from_folder(folder_path):
    """
    从文件夹准备数据，检查必需文件是否存在
    
    参数:
        folder_path: 文件夹路径
        
    返回:
        folder_path: 文件夹路径（验证后）
    """
    print(f"处理数据文件夹: {folder_path}")
    
    # 检查必需的文件是否存在
    required_files = ["train_filled.csv", "eval.csv", "test_filled.csv"]
    for file_name in required_files:
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"缺少必需的文件: {file_path}")
    
    return folder_path


# 新增文件夹数据验证功能
def prepare_data_from_folder(folder_path):
    """
    从文件夹准备数据，检查必需文件是否存在
    
    参数:
        folder_path: 文件夹路径
        
    返回:
        folder_path: 文件夹路径（验证后）
    """
    print(f"处理数据文件夹: {folder_path}")
    
    # 检查必需的文件是否存在
    required_files = ["train_filled.csv", "eval.csv", "test_filled.csv"]
    for file_name in required_files:
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"缺少必需的文件: {file_path}")
    
    return folder_path


# 测试代码
if __name__ == "__main__":
    print("数据处理工具模块已创建")
    print("提供以下功能:")
    print("1. prepare_data_from_files: 根据路径类型处理数据")
    print("2. prepare_data_from_single_csv: 处理单个CSV文件并分割")
    print("3. prepare_data_from_folder: 验证文件夹数据")
    print("3. prepare_data_from_folder: 验证文件夹数据")