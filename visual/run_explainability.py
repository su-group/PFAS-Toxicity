#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可解释性分析工具使用示例

该脚本演示如何使用独立的可解释性分析工具来分析模型预测。
"""

import os
import sys
import torch

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# 动态导入可解释性分析工具
try:
    from visual.explainability_analyzer import GNNExplainerAnalyzer, load_model
    TOOL_AVAILABLE = True
except ImportError as e:
    print(f"导入可解释性分析工具失败: {e}")
    TOOL_AVAILABLE = False


def run_explainability_example():
    """
    运行可解释性分析示例
    """
    if not TOOL_AVAILABLE:
        print("可解释性分析工具不可用")
        return
        
    print("GNNExplainer 可解释性分析示例")
    print("=" * 40)
    
    # 示例SMILES列表
    smiles_list = [
        "CCO",              # 乙醇
        "CC(=O)O",          # 乙酸
        "c1ccccc1",         # 苯
        "CCN(CC)CC",        # 三乙胺
        "C1CCCCC1"          # 环己烷
    ]
    
    # 示例模型参数（根据实际模型调整）
    model_params = {
        'input_dim': 11,      # 原子类型数
        'hidden_dim': 64,
        'num_layers': 3,
        'dropout': 0.2,
        'prediction_tasks': 1
    }
    
    try:
        # 由于我们没有实际模型，创建一个示例模型用于演示
        module = __import__("models.gnn_model", fromlist=["MoleculeGNN"])
        MoleculeGNN = getattr(module, "MoleculeGNN")
        model = MoleculeGNN(**model_params)
        
        # 创建分析器
        analyzer = GNNExplainerAnalyzer(
            model=model,
            model_type="GNN",
            output_dir="./explanation_results"
        )
        
        # 分析多个分子
        print(f"\n开始分析 {len(smiles_list)} 个分子...")
        results = analyzer.analyze_molecule_list(smiles_list, device='cpu')
        
        # 生成摘要报告
        if results:
            analyzer.generate_summary_report(results)
            print(f"\n分析完成，结果已保存到: ./explanation_results")
        else:
            print("分析失败，请检查输入和依赖库")
            
    except Exception as e:
        print(f"运行示例时出错: {e}")


# 提供多种运行模式的函数
def run_full_explanation():
    """运行完整可解释性分析"""
    run_explainability_example()


def run_single_explanation():
    """运行单个分子可解释性分析"""
    run_single_molecule_example()


if __name__ == "__main__":
    # 运行完整示例
    run_full_explanation()
    
    # 运行单个分子示例
    run_single_explanation()