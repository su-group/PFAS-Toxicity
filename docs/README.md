# 分子图神经网络 (GNN) 工具包

该工具包提供了一套完整的分子图神经网络解决方案，包括模型训练、评估、解释等功能。

## 功能模块

### 1. 预训练 (pretrain.py)
执行模型的预训练任务，使用大量无标签数据训练基础模型。

主要特性：
- 支持多种GNN架构 (GNN, GAT, GIN, JKNet, Transformer)
- 可配置的训练参数
- 模型检查点和最佳模型自动保存
- TensorBoard日志记录

### 2. 迁移学习 (transfer_learning.py)
基于预训练模型进行迁移学习，适应特定任务。

主要特性：
- 模型权重迁移
- 层冻结和微调
- 部分迁移支持（形状匹配层迁移）

### 3. 预测与评估 (predict_evaluate.py)
对训练好的模型进行预测和性能评估。

主要特性：
- 单分子性质预测
- 批量分子预测
- 模型性能评估（MSE, MAE, R2等指标）
- 结果可视化

### 4. 半监督学习 (semi_supervised_train.py)
在标签数据稀缺的情况下进行模型训练。

主要特性：
- 使用少量标记数据训练
- 利用图结构传播标签信息
- 支持多种GNN架构

### 5. 模型解释 (explainer.py)
解释模型决策过程，可视化分子中原子的重要性。

主要特性：
- 使用GNNExplainer进行模型解释
- 分子结构可视化
- 原子贡献度分析

## 使用方法

每个模块都可以独立运行，通过修改文件开头的配置参数来调整行为。

### 预训练模型
```bash
python pretrain.py
```

### 迁移学习
```bash
python transfer_learning.py
```

### 预测与评估
```bash
python eval_random.py
```

### 半监督学习
```bash
python semi_supervised_train.py
```

### 模型解释
```bash
python explainer.py
```

## 配置

大多数模块支持通过修改文件开头的配置字典来调整参数。部分模块也支持通过config.json文件进行配置。

## 输出

模型和结果默认保存在`outputs`目录中。