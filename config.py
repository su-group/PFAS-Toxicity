#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置管理模块

该模块提供统一的配置管理功能，包括默认配置、配置加载、验证和保存等
"""

import json
import os
from datetime import datetime


class ConfigManager:
    """配置管理器类"""
    
    # 默认配置
    DEFAULT_CONFIG = {
        "DATA_CONFIG": {
            "data_dir": "data",
            "csv_file": "Labeled-Gap-IsoelectronicSubstitution.csv",
            "smiles_col": "SMILES",
            "label_col": "Gap(eV)",
            "batch_size": 32,
            "test_size": 0.1,
            "val_size": 0.1,
            "random_state": 42,
            "normalize_labels": True,
            "max_atoms": 512
        },
        "MODEL_CONFIG": {
            "default_type": "GNN",
            "supported_types": ["GNN", "GAT", "GIN", "Transformer", "JKNet"],
            "params": {
                "hidden_dim": 256,
                "num_layers": 8,
                "dropout": 0.2,
                "prediction_tasks": 1
            },
            "type_specific_params": {
                "GAT": {
                    "num_heads": 4
                },
                "Transformer": {
                    "num_heads": 4
                },
                "JKNet": {
                    "jk_mode": "cat"
                },
                "GNN": {
                    "model_type": "GCN"
                }
            }
        },
        "TRAINING_CONFIG": {
            "epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 1e-05,
            "checkpoint_interval": 50,
            "device": "cuda"
        },
        "EVALUATION_CONFIG": {
            "metrics": ["mae", "rmse", "r2"]
        },
        "TRANSFER_CONFIG": {
            "pretrained_model_path": "outputs/pretrained_best_model.pth",
            "layers_to_freeze": [],
            "partial_transfer": True
        },
        "SEMI_SUPERVISED_CONFIG": {
            "model_type": "GNN",
            "train_ratio": 0.1,
            "val_ratio": 0.1,
            "epochs": 200,
            "learning_rate": 0.01,
            "weight_decay": 5e-4
        },
        "EXPLAINER_CONFIG": {
            "model_path": "outputs/task_20251009-153211/pretrained_best_model.pth",
            "model_type": "GNN",
            "data_path": None,
            "smiles_list": ["CC(=O)OC1=CC=CC=C1C(=O)O"],
            "save_dir": "explanations",
            "num_epochs": 200
        },
        "PREDICT_EVALUATE_CONFIG": {
            "model_path": None,
            "data_path": None,
            "mode": "predict",
            "smiles_list": [],
            "save_visualization": False,
            "visualization_path": "evaluation_results.png",
            "model_type": "GNN"
        },
        "VISUALIZATION_CONFIG": {
            "figure_dpi": 300,
            "figure_formats": ["png"]
        },
        "LOGGING_CONFIG": {
            "log_level": "INFO"
        }
    }
    
    def __init__(self, config_path=None):
        """
        初始化配置管理器
        
        参数:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config_path = config_path
        if config_path and os.path.exists(config_path):
            self.config = self.load_config(config_path)
        else:
            self.config = self.DEFAULT_CONFIG.copy()
            # 添加时间戳
            self.config["TIMESTAMP"] = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    def load_config(self, config_path):
        """
        从文件加载配置
        
        参数:
            config_path: 配置文件路径
            
        返回:
            config: 配置字典
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            # 添加时间戳（如果不存在）
            if "TIMESTAMP" not in config:
                config["TIMESTAMP"] = datetime.now().strftime("%Y%m%d-%H%M%S")
            return config
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return self.DEFAULT_CONFIG.copy()
    
    def save_config(self, config_path=None):
        """
        保存配置到文件
        
        参数:
            config_path: 配置文件路径，如果为None则使用初始化时的路径
        """
        save_path = config_path or self.config_path
        if not save_path:
            raise ValueError("未指定配置文件保存路径")
            
        try:
            # 创建目录（如果不存在）
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def get(self, section, key=None, default=None):
        """
        获取配置值
        
        参数:
            section: 配置部分名称
            key: 配置键名，如果为None则返回整个部分
            default: 默认值
            
        返回:
            配置值
        """
        if section not in self.config:
            return default
            
        if key is None:
            return self.config[section]
            
        return self.config[section].get(key, default)
    
    def set(self, section, key, value):
        """
        设置配置值
        
        参数:
            section: 配置部分名称
            key: 配置键名
            value: 配置值
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def update_config(self, new_config):
        """
        更新配置
        
        参数:
            new_config: 新的配置字典
        """
        def deep_update(original, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                    deep_update(original[key], value)
                else:
                    original[key] = value
        
        deep_update(self.config, new_config)


# 创建全局配置管理器实例
config_manager = ConfigManager("config.json")


def get_predict_evaluate_config():
    """获取预测评估配置"""
    return config_manager.get("PREDICT_EVALUATE_CONFIG")

# 便捷访问函数
def get_data_config():
    """获取数据配置"""
    return config_manager.get("DATA_CONFIG")


def get_model_config():
    """获取模型配置"""
    return config_manager.get("MODEL_CONFIG")


def get_training_config():
    """获取训练配置"""
    return config_manager.get("TRAINING_CONFIG")


def get_evaluation_config():
    """获取评估配置"""
    return config_manager.get("EVALUATION_CONFIG")


def get_model_params(model_type=None):
    """
    获取模型参数
    
    参数:
        model_type: 模型类型，如果为None则使用默认类型
        
    返回:
        model_params: 模型参数字典
    """
    model_config = get_model_config()
    if model_type is None:
        model_type = model_config.get('default_type', 'GNN')
        
    # 基础参数
    model_params = model_config['params'].copy()
    
    # 添加模型特定参数
    type_specific_params = model_config.get('type_specific_params', {})
    if model_type in type_specific_params:
        model_params.update(type_specific_params[model_type])
        
    return model_params


def get_model_save_path(filename):
    """
    获取模型保存路径
    
    参数:
        filename: 文件名
        
    返回:
        path: 完整路径
    """
    timestamp = config_manager.config.get("TIMESTAMP", "unknown")
    save_dir = os.path.join("outputs", f"task_{timestamp}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, filename)


def get_tensorboard_log_path():
    """
    获取TensorBoard日志路径
    
    返回:
        path: 日志路径
    """
    timestamp = config_manager.config.get("TIMESTAMP", "unknown")
    log_dir = os.path.join("logs", f"task_{timestamp}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_evaluation_path():
    """
    获取评估结果保存路径
    
    返回:
        path: 评估结果保存路径
    """
    timestamp = config_manager.config.get("TIMESTAMP", "unknown")
    eval_dir = os.path.join("outputs", f"task_{timestamp}", "evaluation")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir, exist_ok=True)
    return eval_dir


def get_transfer_config():
    """获取迁移学习配置"""
    return config_manager.get("TRANSFER_CONFIG")


def get_semi_supervised_config():
    """获取半监督学习配置"""
    return config_manager.get("SEMI_SUPERVISED_CONFIG")


def get_explainer_config():
    """获取模型解释配置"""
    return config_manager.get("EXPLAINER_CONFIG")





if __name__ == "__main__":
    print("配置管理模块已创建")
    print("支持以下功能:")
    print("1. ConfigManager: 配置管理器类")
    print("2. get_data_config: 获取数据配置")
    print("3. get_model_config: 获取模型配置")
    print("4. get_training_config: 获取训练配置")
    print("5. get_evaluation_config: 获取评估配置")
    print("6. get_model_params: 获取模型参数")
    print("7. get_model_save_path: 获取模型保存路径")
    print("8. get_tensorboard_log_path: 获取TensorBoard日志路径")
    print("9. get_evaluation_path: 获取评估结果保存路径")
    print(get_predict_evaluate_config())