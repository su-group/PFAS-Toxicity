#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练模块

该模块提供模型训练功能，包括训练循环、模型保存和TensorBoard日志记录
"""

import torch
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 导入上级目录的config模块
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
import config


class MoleculeTrainer:
    """
    分子模型训练器类
    
    该类负责模型的训练过程，包括损失计算、参数更新、模型保存和日志记录
    """

    def __init__(self, model, criterion, optimizer, device):
        """
        初始化训练器
        
        参数:
            model: 待训练的模型
            criterion: 损失函数
            optimizer: 优化器
            device: 计算设备
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.writer = SummaryWriter(log_dir=config.get_tensorboard_log_path())

    def train(self, train_loader, val_loader, epochs=50, save_path=None):
        """
        训练模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            save_path: 最佳模型保存路径（如果为None，则使用配置中的默认路径）
            
        返回:
            tuple: 训练损失和验证损失列表
        """
        if save_path is None:
            save_path = config.get_model_save_path("best_model.pth")
            
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        # 训练进度条
        for epoch in tqdm(range(epochs), position=0, leave=True):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_samples = 0

            for data in train_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                
                # 前向传播
                output, uncertainty = self.model(data.x, data.edge_index, data.batch)
                loss = self.criterion(output, data.y, uncertainty)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 累计损失
                train_loss += loss.item() * data.num_graphs
                train_samples += data.num_graphs

            # 计算平均训练损失
            avg_train_loss = train_loss / train_samples
            train_losses.append(avg_train_loss)

            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_samples = 0

            with torch.no_grad():
                for data in val_loader:
                    data = data.to(self.device)
                    output, uncertainty = self.model(data.x, data.edge_index, data.batch)
                    loss = self.criterion(output, data.y, uncertainty)
                    
                    val_loss += loss.item() * data.num_graphs
                    val_samples += data.num_graphs

            # 计算平均验证损失
            avg_val_loss = val_loss / val_samples
            val_losses.append(avg_val_loss)

            # TensorBoard日志记录
            self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

            # 更新最佳验证损失并保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss
                }, save_path)
                print(f"在第 {epoch+1} 轮保存了最佳模型，验证损失: {avg_val_loss:.4f}")

            # 每隔一定轮数保存检查点
            if (epoch + 1) % config.TRAINING_CONFIG['checkpoint_interval'] == 0:
                checkpoint_path = config.get_model_save_path(f"checkpoint_epoch_{epoch+1}.pth")
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss
                }, checkpoint_path)

        # 关闭TensorBoard写入器
        self.writer.close()
        
        print(f"\n训练完成! 最佳验证损失: {best_val_loss:.6f}")
        return train_losses, val_losses

    def finetune(self, train_loader, val_loader, epochs=20, save_path=None, unfreeze_all=False):
        """
        微调模型（迁移学习）
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 微调轮数
            save_path: 微调模型保存路径
            unfreeze_all: 是否解冻所有层进行微调
            
        返回:
            tuple: 训练损失和验证损失列表
        """
        if save_path is None:
            save_path = config.get_model_save_path("finetuned_model.pth")
            
        # 如果选择解冻所有层，则设置所有参数为可训练
        if unfreeze_all:
            for param in self.model.parameters():
                param.requires_grad = True
            print("已解冻所有层进行微调")
            
        # 使用较小的学习率进行微调
        original_lr = self.optimizer.param_groups[0]['lr']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = original_lr * 0.1  # 微调时使用更小的学习率
            
        print(f"使用学习率 {original_lr * 0.1} 进行微调")
        
        # 执行训练
        train_losses, val_losses = self.train(train_loader, val_loader, epochs, save_path)
        
        # 恢复原始学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = original_lr
            
        return train_losses, val_losses

    def get_feature_importance(self, data):
        """
        计算特征重要性（原子贡献度）

        参数:
            data: PyG数据对象

        返回:
            importance: 特征重要性张量
        """
        # 确保数据在正确的设备上
        data = data.to(self.device)
        
        # 启用梯度追踪
        data.x.requires_grad_(True)

        # 前向传播
        predictions, uncertainties = self.model(data.x, data.edge_index, data.batch)

        # 计算梯度
        prediction_loss = predictions.sum()
        prediction_loss.backward()

        # 计算L2范数作为重要性
        importance = torch.norm(data.x.grad, dim=1, p=2)

        # 清除梯度
        data.x.requires_grad_(False)

        return importance.detach()


# 测试代码
if __name__ == "__main__":
    print("训练模块已创建")
    print("请使用完整训练脚本进行模型训练")