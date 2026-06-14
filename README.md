# Unveiling the Structural Determinants of PFAS Toxicity

## Paper Information

**Paper Title**: *Unveiling the Structural Determinants of PFAS Toxicity: A Graph Neural Network and Interpretability Analysis*

**Project Description**: This repository contains the code implementation for the above academic paper, designed to reproduce the experimental results presented in the paper.

---

## Project Overview

This project uses Graph Neural Networks (GNNs) for toxicity prediction and interpretability analysis of Per- and Polyfluoroalkyl Substances (PFAS). Through a two-stage training strategy of pre-training followed by transfer learning, combined with multiple GNN architectures and GNNExplainer interpretability methods, we systematically reveal the relationship between PFAS molecular structure and toxicity.

### Research Objectives

1. **Toxicity Prediction**: Predict acute toxicity of PFAS molecules (represented as negative logarithm of LD50) using GNN models
2. **Model Comparison**: Compare the performance of multiple GNN architectures (GCN, GAT, GIN, JKNet, Transformer)
3. **Interpretability Analysis**: Identify the structural units and functional groups that contribute most to toxicity using GNNExplainer

### Core Methods

- **Pre-training Stage**: Pre-train on large-scale LD50 molecular dataset (LDToxDB)
- **Transfer Learning Stage**: Fine-tune pre-trained models on PFAS-specific datasets
- **Data Splitting Strategies**: Random Split and Start Split
- **Interpretability Method**: GNNExplainer for atomic contribution and functional group attribution analysis

---

## Directory Structure

```
GNN_code/
├── Scripts/              # Main training and evaluation scripts
│   ├── pretrain.py                    # Pre-training script
│   ├── transfer_learning.py           # Transfer learning script
│   ├── eval_pretrain.py               # Evaluate pre-trained model
│   ├── eval_revision_260527.py        # Evaluation script (revision)
│   └── predict_transfer_learning.py   # Prediction script
├── models/               # Model definitions
│   ├── gnn_model.py                   # Basic GNN (GCN/SAGE)
│   ├── gat_model.py                   # GAT model
│   ├── gin_model.py                   # GIN model
│   ├── jknet_model.py                 # JKNet model
│   └── transformer_model.py           # Graph Transformer
├── dataset/              # Dataset modules
│   ├── molecule_dataset.py            # Molecule dataset class
│   ├── graph_builder.py               # Graph structure builder
│   └── data_utils.py                  # Data utility functions
├── data/                 # Data directory
│   ├── cf2_data/                     # CF2 summary data
│   ├── cf2_data_rand/                # Random split data
│   │   ├── train/                    # Training set
│   │   ├── val/                      # Validation set
│   │   └── test/                     # Test set
│   ├── cf2_data_start/               # Start split data
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── PFAS_id50.csv                 # PFAS dataset
│   └── ldtoxdb-raw.csv               # LDToxDB pre-training data
├── outputs/              # Model outputs
│   ├── GIN/                          # GIN model outputs
│   ├── GNN/                          # GNN (GCN) model outputs
│   └── GAT/                          # GAT model outputs
├── train/                # Training modules
├── test/                 # Test modules
├── visual/               # Visualization modules
├── explainers/           # Interpretability scripts
│   ├── exp_sar_251224.py             # SAR analysis
│   ├── exp_sar_proprity.py           # Property analysis
│   ├── functional_group_attribution_analysis.py  # Functional group attribution
│   ├── interact_experiments_groupid_251125.py   # Interactive experiments
│   ├── experiment3_figure11.py       # Figure 11 generation
│   └── experiment3_figure12.py       # Figure 12 generation
├── data_process_class/   # Data processing & analysis
├── config.py             # Configuration manager
├── config.json           # Configuration file
├── utils.py              # Utility functions
├── README.md             # This file
└── README.zh-CN.md    # Chinese version | 中文版
```

---

## Requirements

### Software Dependencies

| Dependency | Version | Description |
|-----------|---------|-------------|
| Python | >= 3.8 | Programming language |
| PyTorch | >= 1.9 | Deep learning framework |
| PyTorch Geometric | >= 2.0 | GNN library |
| RDKit | - | Cheminformatics library |
| pandas | - | Data processing |
| numpy | - | Numerical computing |
| scikit-learn | - | Machine learning tools |
| tensorboard | - | Training visualization |
| tqdm | - | Progress bar |
| matplotlib | - | Plotting (for figures) |
| seaborn | - | Statistical plotting (optional) |

### Hardware Requirements

- **Recommended GPU**: NVIDIA GPU with CUDA support (VRAM >= 8GB)
- **CPU Mode**: Can also run on CPU, but training will take longer

---

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
conda create -n pfas_gnn python=3.9
conda activate pfas_gnn
```

### 2. Install PyTorch

Choose the appropriate installation command based on your CUDA version:

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install PyTorch Geometric

```bash
pip install torch_geometric

# Install CUDA extensions (choose based on PyTorch and CUDA versions)
# Reference: https://data.pyg.org/whl/
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 4. Install Other Dependencies

```bash
pip install rdkit pandas numpy scikit-learn tensorboard tqdm matplotlib
```

---

## Reproducing Paper Experiments

The following steps will guide you through reproducing the main experimental results from the paper.

### Stage 1: Model Pre-training

Pre-train on the LDToxDB dataset:

```bash
cd Scripts
python pretrain.py
```

**Notes**:
- Pre-training data: `ldtoxdb-raw.csv` (large-scale LD50 molecular dataset)
- Model type: GIN (best performing model in the paper)
- Training epochs: 1000 epochs
- Output directory: `outputs/GIN_<timestamp>/`

### Stage 2: Transfer Learning

Transfer pre-trained models to the PFAS dataset:

```bash
# Random Split
python transfer_learning.py

# Or use specialized transfer learning script
python transfer_learning_spilts_251107.py
```

**Notes**:
- Transfer data: `PFAS_id50.csv` or processed PFAS datasets
- Two splitting strategies supported:
  - **Random Split**: Randomly split train/val/test sets
  - **Start Split**: Split by time/structural order (more challenging)
- Output directory: `outputs/GIN_<timestamp>/GIN_TL_<strategy>_<timestamp>/`

### Model Evaluation

Evaluate trained models:

```bash
# Evaluate pre-trained model
python eval_pretrain.py

# Evaluate revised model (for final paper results)
python eval_revision_260527.py
```

### Interpretability Analysis

Run model explanation and visualization scripts:

```bash
cd ../explainers

# SAR (Structure-Activity Relationship) analysis
python exp_sar_251224.py

# Functional group attribution analysis
python functional_group_attribution_analysis.py

# Generate figures from the paper
python experiment3_figure11.py
python experiment3_figure12.py
```

**Analysis Content**:
- Atomic contribution visualization
- Functional group toxicity contribution ranking
- PFAS molecular structure feature analysis

---

## Configuration Guide

Main configurations are managed through the [config.json](file:///c:/Users/test/Downloads/Revised_260601_zys/GNN_code/config.json) file.

### Data Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data_dir` | Data directory | `"data"` |
| `csv_file` | Data filename | `"ldtoxdb-raw.csv"` |
| `smiles_col` | SMILES column name | `"SMILES"` |
| `label_col` | Label column name (NeglogLD50) | `"NeglogLD50"` |
| `batch_size` | Batch size | `32` |
| `test_size` | Test set ratio | `0.1` |
| `val_size` | Validation set ratio | `0.1` |
| `normalize_labels` | Whether to normalize labels | `true` |

### Model Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_type` | Model type | `"GIN"` |
| `hidden_dim` | Hidden layer dimension | `256` |
| `num_layers` | Number of GNN layers | `8` |
| `dropout` | Dropout rate | `0.2` |
| `prediction_tasks` | Number of prediction tasks | `1` |

**Supported Model Types**:
- `GNN`: Basic GNN based on GCN
- `GAT`: Graph Attention Network
- `GIN`: Graph Isomorphism Network (recommended in paper)
- `JKNet`: Jumping Knowledge Network
- `Transformer`: Graph Transformer

### Training Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `epochs` | Number of pre-training epochs | `1000` |
| `learning_rate` | Learning rate | `0.001` |
| `weight_decay` | Weight decay | `1e-5` |
| `checkpoint_interval` | Checkpoint save interval | `10` |
| `device` | Computing device | `"cuda"` |

### Transfer Learning Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `pretrained_model_path` | Path to pre-trained model | `"outputs/pretrained_best_model.pth"` |
| `layers_to_freeze` | Layers to freeze | `[]` (none) |
| `partial_transfer` | Partial transfer | `true` |

---

## Model Architecture

### Uncertainty Estimation

All models integrate an uncertainty prediction head, trained with `UncertaintyLoss`:

```python
from models.gnn_model import UncertaintyLoss

criterion = UncertaintyLoss(alpha=0.1)
```

**Loss Function Components**:
1. **Prediction Loss**: MSE Loss (Mean Squared Error)
2. **Uncertainty Loss**: Consistency between predicted uncertainty and actual error

### Model Comparison

| Model | Characteristics | Paper Performance |
|-------|-----------------|-------------------|
| GIN | Graph Isomorphism Network, strongest expressive power | Best (Recommended) |
| GNN (GCN) | Graph Convolutional Network, basic model | Good |
| GAT | Graph Attention Network, interpretable | Good |
| JKNet | Multi-layer feature fusion | Good |
| Transformer | Long-range dependency modeling | Moderate |

---

## Output Structure

After training, models and results are saved in the `outputs/` directory:

```
outputs/
├── GIN_20260107-094723/           # GIN pre-training output
│   ├── pretrained_best_model.pth      # Best validation model
│   ├── pretrained_final_model.pth     # Final model
│   ├── evaluation_report.csv          # Evaluation report
│   └── GIN_TL_Start_cf2_350_task_20260528-184435/  # Transfer learning output
│       ├── transfer_learned_best_model.pth
│       ├── transfer_learned_final_model.pth
│       └── transfer_checkpoint_epoch_350.pth
├── GNN/                            # GNN (GCN) model outputs
│   └── GNN_pretrain_20251105-191945/
└── GAT/                            # GAT model outputs
    └── GAT_pretrain_20260106-131012/
```

---

## TensorBoard Visualization

Training loss curves are logged to TensorBoard:

```bash
tensorboard --logdir logs
```

Then visit `http://localhost:6006` in your browser to view training curves.

---

## Main Module Documentation

### Configuration Manager

**File**: [config.py](file:///c:/Users/test/Downloads/Revised_260601_zys/GNN_code/config.py)

The `ConfigManager` class provides unified configuration management:

```python
from config import get_model_config, get_training_config

model_config = get_model_config()
training_config = get_training_config()
```

### Dataset

**File**: [dataset/molecule_dataset.py](file:///c:/Users/test/Downloads/Revised_260601_zys/GNN_code/dataset/molecule_dataset.py)

The `MoleculeDataset` class automatically handles SMILES to graph conversion:

```python
from dataset.molecule_dataset import MoleculeDataset

dataset = MoleculeDataset(
    root="data",
    filename="ldtoxdb-raw.csv",
    smiles_col="SMILES",
    label_col="NeglogLD50"
)
```

**Features**:
- Automatically convert SMILES to molecular graphs
- Support label normalization
- Auto-cache processed data

### Utility Functions

**File**: [utils.py](file:///c:/Users/test/Downloads/Revised_260601_zys/GNN_code/utils.py)

Provides model saving/loading, TensorBoard logging, and other utilities.

---

## Data Documentation

### Datasets

| Dataset | Filename | Purpose |
|---------|----------|---------|
| LDToxDB | `ldtoxdb-raw.csv` | Pre-training (large-scale LD50 data) |
| PFAS | `PFAS_id50.csv` | Transfer learning (PFAS-specific) |
| CF2+ | `ldtoxdb-cf2plus.csv` | CF2-enhanced dataset |

### Data Format

CSV files must contain the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| SMILES | SMILES representation of molecule | `C(F)(F)(S(O)(=O)=O)C(F)(F)C(F)(F)F` |
| NeglogLD50 | Toxicity label (-log(LD50)) | `4.52` |

---

## Interpretability Analysis

### Analysis Scripts

The `explainers/` directory contains the following analysis scripts:

| Script | Function |
|--------|----------|
| `exp_sar_251224.py` | SAR (Structure-Activity Relationship) analysis |
| `exp_sar_proprity.py` | Molecular property-toxicity correlation |
| `functional_group_attribution_analysis.py` | Functional group attribution |
| `interact_experiments_groupid_251125.py` | Interactive experiments |
| `experiment3_figure11.py` | Generate Figure 11 |
| `experiment3_figure12.py` | Generate Figure 12 |

### Analysis Content

1. **Atomic Contributions**: Use GNNExplainer to identify atoms contributing most to toxicity prediction
2. **Functional Group Attribution**: Analyze impact of common functional groups on PFAS toxicity
3. **Structure-Activity Relationship**: Establish relationships between structural features and toxicity
4. **Visualization**: Generate figures for the paper

---

## FAQ

**Q: How to switch between different GNN models?**

A: Modify `MODEL_CONFIG.model_type` in [config.json](file:///c:/Users/test/Downloads/Revised_260601_zys/GNN_code/config.json) or directly edit the configuration in scripts.

**Q: How to switch between CPU/GPU training?**

A: Modify `TRAINING_CONFIG.device` to `"cpu"` or `"cuda"`.

**Q: Where to configure the pre-trained model path?**

A: Configure `TRANSFER_CONFIG.pretrained_model_path`.

**Q: How to adjust the data splitting strategy?**

A: Modify the data loading logic in `transfer_learning.py`, or use different dataset directories (`cf2_data_rand` vs `cf2_data_start`).

---

## Citation

If you use this code in your research, please cite the following paper:

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

## License

This project is for academic research use only.

---

## Contact

For questions, please contact through the paper's contact information or submit an Issue.
