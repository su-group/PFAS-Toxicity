# 揭示全氟和多氟烷基物质毒性的结构决定因素

## 论文信息

**论文标题**：*Unveiling the Structural Determinants of PFAS Toxicity: A Graph Neural Network and Interpretability Analysis*

**项目说明**：本代码仓库是上述学术论文的配套实现，用于复现论文中的实验结果。

> **English Version**: 请查看 [README.md](file:///c:/Users/test/Downloads/Revised_260601_zys/GNN_code/README.md) 查看英文版本。

---

## 项目概述

本项目利用图神经网络（Graph Neural Networks, GNN）对全氟和多氟烷基物质（PFAS）的毒性进行预测和可解释性分析。通过预训练-迁移学习的两阶段训练策略，结合多种 GNN 架构和 GNNExplainer 可解释性方法，系统揭示了 PFAS 分子结构与毒性之间的关系。

### 研究目标

1. **毒性预测**：使用 GNN 模型预测 PFAS 分子的急性毒性（以 LD50 的负对数表示）
2. **模型比较**：比较多种 GNN 架构（GCN、GAT、GIN、JKNet、Transformer）的性能
3. **可解释性分析**：通过 GNNExplainer 识别对毒性贡献最大的结构单元和官能团

### 核心方法

- **预训练阶段**：在大规模 LD50 分子数据集（LDToxDB）上进行预训练
- **迁移学习阶段**：将预训练模型迁移到 PFAS 特异性数据集进行微调
- **数据划分策略**：随机划分（Random Split）和起始划分（Start Split）
- **可解释性方法**：GNNExplainer 进行原子贡献度和官能团归因分析

---

## 目录结构

```
GNN_code/
├── Scripts/              # 主要训练和评估脚本
│   ├── pretrain.py                    # 预训练脚本
│   ├── transfer_learning.py           # 迁移学习脚本
│   ├── eval_pretrain.py               # 预训练模型评估
│   ├── eval_revision_260527.py        # 修订版评估脚本
│   └── predict_transfer_learning.py   # 预测脚本
├── models/               # 模型定义
│   ├── gnn_model.py                   # 基础 GNN (GCN/SAGE)
│   ├── gat_model.py                   # GAT 模型
│   ├── gin_model.py                   # GIN 模型
│   ├── jknet_model.py                 # JKNet 模型
│   └── transformer_model.py           # Graph Transformer
├── dataset/              # 数据集模块
│   ├── molecule_dataset.py            # 分子数据集类
│   ├── graph_builder.py               # 图结构构建
│   └── data_utils.py                  # 数据工具函数
├── data/                 # 数据目录
│   ├── cf2_data/                     # CF2 汇总数据
│   ├── cf2_data_rand/                # 随机划分数据 (Rand)
│   │   ├── train/                    # 训练集
│   │   ├── val/                      # 验证集
│   │   └── test/                     # 测试集
│   ├── cf2_data_start/               # 起始划分数据 (Start)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── PFAS_id50.csv                 # PFAS 数据集
│   └── ldtoxdb-raw.csv               # LDToxDB 预训练数据
├── outputs/              # 模型输出目录
│   ├── GIN/                          # GIN 模型输出
│   ├── GNN/                          # GNN (GCN) 模型输出
│   └── GAT/                          # GAT 模型输出
├── train/                # 训练相关模块
├── test/                 # 测试模块
├── visual/               # 可视化模块
├── explainers/           # 可解释性分析脚本
│   ├── exp_sar_251224.py             # SAR 分析实验
│   ├── exp_sar_proprity.py           # 性质分析
│   ├── functional_group_attribution_analysis.py  # 官能团归因分析
│   ├── interact_experiments_groupid_251125.py   # 交互式实验
│   ├── experiment3_figure11.py       # 图 11 生成脚本
│   └── experiment3_figure12.py       # 图 12 生成脚本
├── data_process_class/   # 数据处理和分析
├── config.py             # 配置管理模块
├── config.json           # 配置文件
├── utils.py              # 工具函数
├── README.md             # 英文版本
└── README.zh-CN.md       # 本文件（中文版）
```

---

## 环境要求

### 软件依赖

| 依赖 | 版本 | 说明 |
|-----|------|------|
| Python | >= 3.8 | 编程语言 |
| PyTorch | >= 1.9 | 深度学习框架 |
| PyTorch Geometric | >= 2.0 | 图神经网络库 |
| RDKit | - | 分子化学信息学库 |
| pandas | - | 数据处理 |
| numpy | - | 数值计算 |
| scikit-learn | - | 机器学习工具 |
| tensorboard | - | 训练可视化 |
| tqdm | - | 进度条 |
| matplotlib | - | 绘图（用于生成图表） |
| seaborn | - | 统计绘图（可选） |

### 硬件要求

- **推荐 GPU**：NVIDIA GPU with CUDA 支持（显存 >= 8GB）
- **CPU 模式**：也可在 CPU 上运行，但训练时间较长

---

## 安装

### 1. 创建虚拟环境（推荐）

```bash
conda create -n pfas_gnn python=3.9
conda activate pfas_gnn
```

### 2. 安装 PyTorch

根据你的 CUDA 版本选择合适的安装命令：

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. 安装 PyTorch Geometric

```bash
pip install torch_geometric

# 安装 CUDA 扩展包（根据 PyTorch 和 CUDA 版本选择）
# 参考: https://data.pyg.org/whl/
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 4. 安装其他依赖

```bash
pip install rdkit pandas numpy scikit-learn tensorboard tqdm matplotlib
```

---

## 复现论文实验

以下步骤将指导您复现论文中的主要实验结果。

### 阶段一：模型预训练

在 LDToxDB 数据集上进行预训练：

```bash
cd Scripts
python pretrain.py
```

**说明**：
- 预训练数据：`ldtoxdb-raw.csv`（大规模 LD50 分子数据集）
- 模型类型：GIN（论文中表现最优的模型）
- 训练轮数：1000 epochs
- 输出目录：`outputs/GIN_<timestamp>/`

### 阶段二：迁移学习

将预训练模型迁移到 PFAS 数据集：

```bash
# 随机划分 (Random Split)
python transfer_learning.py

# 或使用专用迁移学习脚本
python transfer_learning_spilts_251107.py
```

**说明**：
- 迁移数据：`PFAS_id50.csv` 或处理后的 PFAS 数据集
- 支持两种划分策略：
  - **Random Split**：随机划分训练/验证/测试集
  - **Start Split**：按时间/结构顺序划分（更具挑战性）
- 输出目录：`outputs/GIN_<timestamp>/GIN_TL_<strategy>_<timestamp>/`

### 模型评估

评估训练好的模型：

```bash
# 评估预训练模型
python eval_pretrain.py

# 评估修订版模型（用于论文最终结果）
python eval_revision_260527.py
```

### 可解释性分析

运行模型解释和可视化脚本：

```bash
cd ../explainers

# SAR 分析（结构-活性关系）
python exp_sar_251224.py

# 官能团归因分析
python functional_group_attribution_analysis.py

# 生成论文中的图表
python experiment3_figure11.py
python experiment3_figure12.py
```

**分析内容**：
- 原子贡献度可视化
- 官能团对毒性的贡献排名
- PFAS 分子结构特征分析

---

## 配置说明

主要配置通过 [config.json](file:///c:/Users/test/Downloads/Revised_260601_zys/GNN_code/config.json) 文件管理。

### 数据配置

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `data_dir` | 数据目录 | `"data"` |
| `csv_file` | 数据文件名 | `"ldtoxdb-raw.csv"` |
| `smiles_col` | SMILES 列名 | `"SMILES"` |
| `label_col` | 标签列名 (NeglogLD50) | `"NeglogLD50"` |
| `batch_size` | 批次大小 | `32` |
| `test_size` | 测试集比例 | `0.1` |
| `val_size` | 验证集比例 | `0.1` |
| `normalize_labels` | 是否标准化标签 | `true` |

### 模型配置

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `model_type` | 模型类型 | `"GIN"` |
| `hidden_dim` | 隐藏层维度 | `256` |
| `num_layers` | GNN 层数 | `8` |
| `dropout` | Dropout 率 | `0.2` |
| `prediction_tasks` | 预测任务数 | `1` |

**支持的模型类型**：
- `GNN`：基于 GCN 的基础 GNN
- `GAT`：图注意力网络
- `GIN`：图同构网络（论文推荐）
- `JKNet`：Jumping Knowledge Network
- `Transformer`：Graph Transformer

### 训练配置

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `epochs` | 预训练轮数 | `1000` |
| `learning_rate` | 学习率 | `0.001` |
| `weight_decay` | 权重衰减 | `1e-5` |
| `checkpoint_interval` | 检查点间隔 | `10` |
| `device` | 计算设备 | `"cuda"` |

### 迁移学习配置

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `pretrained_model_path` | 预训练模型路径 | `"outputs/pretrained_best_model.pth"` |
| `layers_to_freeze` | 冻结的层 | `[]`（不冻结） |
| `partial_transfer` | 部分迁移 | `true` |

---

## 模型架构

### 不确定性估计

所有模型集成了不确定性预测头，使用 `UncertaintyLoss` 进行训练：

```python
from models.gnn_model import UncertaintyLoss

criterion = UncertaintyLoss(alpha=0.1)
```

**损失函数组成**：
1. **预测损失**：MSE Loss（均方误差）
2. **不确定性损失**：预测不确定性与实际误差的一致性

### 模型对比

| 模型 | 特点 | 论文中的表现 |
|-----|------|-------------|
| GIN | 图同构网络，表达能力最强 | 最优（推荐） |
| GNN (GCN) | 图卷积网络，基础模型 | 良好 |
| GAT | 图注意力网络，可解释性 | 良好 |
| JKNet | 多层特征融合 | 良好 |
| Transformer | 长程依赖建模 | 一般 |

---

## 输出结构

训练完成后，模型和结果保存在 `outputs/` 目录：

```
outputs/
├── GIN_20260107-094723/           # GIN 预训练输出
│   ├── pretrained_best_model.pth      # 最佳验证模型
│   ├── pretrained_final_model.pth     # 最终模型
│   ├── evaluation_report.csv          # 评估报告
│   └── GIN_TL_Start_cf2_350_task_20260528-184435/  # 迁移学习输出
│       ├── transfer_learned_best_model.pth
│       ├── transfer_learned_final_model.pth
│       └── transfer_checkpoint_epoch_350.pth
├── GNN/                            # GNN (GCN) 模型输出
│   └── GNN_pretrain_20251105-191945/
└── GAT/                            # GAT 模型输出
    └── GAT_pretrain_20260106-131012/
```

---

## TensorBoard 可视化

训练过程中的损失曲线会记录到 TensorBoard：

```bash
tensorboard --logdir logs
```

然后在浏览器中访问 `http://localhost:6006` 查看训练曲线。

---

## 主要模块说明

### 配置管理

**文件**：[config.py](file:///c:/Users/test/Downloads/Revised_260601_zys/GNN_code/config.py)

`ConfigManager` 类提供统一的配置管理：

```python
from config import get_model_config, get_training_config

model_config = get_model_config()
training_config = get_training_config()
```

### 数据集

**文件**：[dataset/molecule_dataset.py](file:///c:/Users/test/Downloads/Revised_260601_zys/GNN_code/dataset/molecule_dataset.py)

`MoleculeDataset` 类自动处理 SMILES 到图的转换：

```python
from dataset.molecule_dataset import MoleculeDataset

dataset = MoleculeDataset(
    root="data",
    filename="ldtoxdb-raw.csv",
    smiles_col="SMILES",
    label_col="NeglogLD50"
)
```

**功能**：
- 自动将 SMILES 转换为分子图
- 支持标签标准化
- 自动缓存处理后的数据

### 工具函数

**文件**：[utils.py](file:///c:/Users/test/Downloads/Revised_260601_zys/GNN_code/utils.py)

提供模型保存/加载、TensorBoard 日志等功能。

---

## 数据说明

### 数据集

| 数据集 | 文件名 | 用途 |
|-------|--------|------|
| LDToxDB | `ldtoxdb-raw.csv` | 预训练（大规模 LD50 数据） |
| PFAS | `PFAS_id50.csv` | 迁移学习（PFAS 特异性数据） |
| CF2+ | `ldtoxdb-cf2plus.csv` | CF2 增强数据集 |

### 数据格式

CSV 文件需要包含以下列：

| 列名 | 说明 | 示例 |
|-----|------|------|
| SMILES | 分子的 SMILES 表示 | `C(F)(F)(S(O)(=O)=O)C(F)(F)C(F)(F)F` |
| NeglogLD50 | 毒性标签（-log(LD50)） | `4.52` |

---

## 可解释性分析

### 分析脚本

`explainers/` 目录包含以下分析脚本：

| 脚本 | 功能 |
|-----|------|
| `exp_sar_251224.py` | SAR（结构-活性关系）分析 |
| `exp_sar_proprity.py` | 分子性质与毒性关联分析 |
| `functional_group_attribution_analysis.py` | 官能团归因分析 |
| `interact_experiments_groupid_251125.py` | 交互式实验分析 |
| `experiment3_figure11.py` | 生成论文图 11 |
| `experiment3_figure12.py` | 生成论文图 12 |

### 分析内容

1. **原子贡献度**：使用 GNNExplainer 识别对毒性预测贡献最大的原子
2. **官能团归因**：分析常见官能团对 PFAS 毒性的影响
3. **结构-活性关系**：建立分子结构特征与毒性之间的关联
4. **可视化**：生成论文中的图表

---

## 常见问题

**Q: 如何切换不同的 GNN 模型？**

A: 修改 [config.json](file:///c:/Users/test/Downloads/Revised_260601_zys/GNN_code/config.json) 中的 `MODEL_CONFIG.model_type` 或直接编辑脚本中的配置。

**Q: 如何切换 CPU/GPU 训练？**

A: 修改 `TRAINING_CONFIG.device` 参数为 `"cpu"` 或 `"cuda"`。

**Q: 预训练模型路径在哪里配置？**

A: 在 `TRANSFER_CONFIG.pretrained_model_path` 中配置。

**Q: 如何调整数据划分策略？**

A: 修改 `transfer_learning.py` 中的数据加载逻辑，或使用不同的数据集目录（`cf2_data_rand` vs `cf2_data_start`）。

---

## 引用

如果您在研究中使用了本代码，请引用以下论文：

```bibtex
@article{xxx,
    title={Unveiling the Structural Determinants of PFAS Toxicity: A Graph Neural Network and Interpretability Analysis},
    author={...},
    journal={...},
    year={2026},
    doi={...}
}
```

---

## 许可证

本项目仅供学术研究使用。

---

## 联系方式

如有问题，请通过论文中的联系方式或提交 Issue 进行咨询。
