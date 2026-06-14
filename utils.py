#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具模块

该模块提供模型存储、加载和数据记录等通用功能
"""

import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置模块
from config import get_tensorboard_log_path


def get_model_save_path(filename):
    """
    获取模型保存路径
    
    参数:
        filename: 文件名
        
    返回:
        path: 完整路径
    """
    save_dir = "outputs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, filename)


def save_model(model, model_type, model_params, save_path, epoch=None, 
               train_loss=None, val_loss=None, optimizer_state_dict=None):
    """
    保存模型
    
    参数:
        model: 模型实例
        model_type: 模型类型
        model_params: 模型参数
        save_path: 保存路径
        epoch: 训练轮数
        train_loss: 训练损失
        val_loss: 验证损失
        optimizer_state_dict: 优化器状态字典
    """
    # 创建保存目录（如果不存在）
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # 准备保存的数据
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'model_params': model_params
    }
    
    # 添加可选信息
    if epoch is not None:
        save_dict['epoch'] = epoch
    if train_loss is not None:
        save_dict['train_loss'] = train_loss
    if val_loss is not None:
        save_dict['val_loss'] = val_loss
    if optimizer_state_dict is not None:
        save_dict['optimizer_state_dict'] = optimizer_state_dict
    
    # 保存模型
    torch.save(save_dict, save_path)
    print(f"模型已保存到: {save_path}")


def load_model(model_class, model_params, model_path, device='cpu'):
    """
    加载模型
    
    参数:
        model_class: 模型类
        model_params: 模型参数
        model_path: 模型路径
        device: 设备
        
    返回:
        model: 加载的模型实例
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 创建模型实例
    model = model_class(**model_params)
    model = model.to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"成功加载模型: {model_path}")
    return model


def create_tensorboard_writer():
    """
    创建TensorBoard写入器
    
    返回:
        writer: TensorBoard写入器实例
    """
    log_dir = get_tensorboard_log_path()
    writer = SummaryWriter(log_dir=log_dir)
    return writer


def log_config_to_tensorboard(writer, config_dict, config_name):
    """
    将配置信息记录到TensorBoard
    
    参数:
        writer: TensorBoard写入器
        config_dict: 配置字典
        config_name: 配置名称
    """
    writer.add_text(f'Config/{config_name}', str(config_dict))


def log_losses_to_tensorboard(writer, train_loss, val_loss, epoch):
    """
    将损失记录到TensorBoard
    
    参数:
        writer: TensorBoard写入器
        train_loss: 训练损失
        val_loss: 验证损失
        epoch: 训练轮数
    """
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)


def close_tensorboard_writer(writer):
    """
    关闭TensorBoard写入器
    
    参数:
        writer: TensorBoard写入器
    """
    writer.close()