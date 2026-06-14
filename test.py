#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本，用于打印从配置文件中获取的实际参数
"""

import sys
import os
from pprint import pprint

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# 导入配置管理模块
from config import get_explainer_config, get_data_config, get_model_config, get_training_config

def test_config():
    """测试并打印配置参数"""
    print("=" * 50)
    print("测试配置参数")
    print("=" * 50)
    
    # 获取解释器配置
    print("\n1. EXPLAINER_CONFIG:")
    print("-" * 30)
    explainer_config = get_explainer_config()
    pprint(explainer_config)
    
    # 获取数据配置
    print("\n2. DATA_CONFIG:")
    print("-" * 30)
    data_config = get_data_config()
    pprint(data_config)
    
    # 获取模型配置
    print("\n3. MODEL_CONFIG:")
    print("-" * 30)
    model_config = get_model_config()
    pprint(model_config)
    
    # 获取训练配置
    print("\n4. TRAINING_CONFIG:")
    print("-" * 30)
    training_config = get_training_config()
    pprint(training_config)

if __name__ == "__main__":
    test_config()