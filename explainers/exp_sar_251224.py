#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced GNN Explanation Experiments with Bug Fixes
保持原有实验二、三，修复实验一的错误
"""

import os
import sys
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor, Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import warnings
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import io
from scipy.stats import pearsonr
import itertools
from typing import List, Dict, Tuple, Any

warnings.filterwarnings('ignore')

# ==================== Configuration ====================
DATA_PATH = r'F:\GNN-pro\data_process_class\step3_OECD_Class_Enhanced_v2_with_CF.csv'
PREDICTIONS_CACHE_PATH = r'F:\GNN-pro\data_process_class\predict_oecd_cf2_GIN510_v3.csv'
PRETRAINED_MODEL_PATH = r"F:\GNN-pro\scripts\outputs\GIN_20260107-094723\GIN_TL_Start_cf2_510_task_20260528-184435\transfer_learned_best_model.pth"

# Model configuration
MODEL_CONFIG = {
    "model_type": "GIN",
    "supported_types": ["GNN", "GAT", "GIN", "Transformer", "JKNet"],
    "params": {
        "hidden_dim": 256,
        "num_layers": 8,
        "dropout": 0.2,
        "prediction_tasks": 1,
        "input_dim": 120,
    },
    "type_specific_params": {
        "GNN": {"model_type": "GCN"},
        "GAT": {"num_heads": 4},
        "JKNet": {"jk_mode": "cat"},
        "Transformer": {"num_heads": 8}
    }
}

# Normalization parameters
TARGET_MEAN = 2.9348
TARGET_STD = 1.0857
NUM_EPOCHS_FOR_EXPLANATION = 50

# ==================== Import existing modules ====================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.graph_builder import MoleculeGraphBuilder
from models.gnn_model import MoleculeGNN
from models.gat_model import MoleculeGAT
from models.gin_model import MoleculeGIN
from models.jknet_model import MoleculeJKNet
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score

# ==================== Data Loading ====================
print("Loading data...")
df_enhanced = pd.read_csv(DATA_PATH)
df_enhanced['total_CF'] = df_enhanced['CF2'] + df_enhanced['CF3'] + df_enhanced['CFX']
print(f"Data loaded, total {len(df_enhanced)} rows.")

# ==================== Functional Group SMARTS Patterns ====================
FUNCTIONAL_GROUP_SMARTS = {
    "CF3": ("[#6](F)(F)F", "Halogen"),
    "CF2": ("[#6](F)F", "Halogen"),
    "I": ("[I]", "Halogen"),
    "OH": ("[OH]", "Polar"),
    "COOH": ("C(=O)[OH]", "Polar"),
    "C=O": ("C=O", "Polar"),
    "O": ("[O]", "Polar"),
    "NH2": ("[N;H2]", "Polar"),
    "SH": ("[SH]", "Polar"),
    "SO3H": ("S(=O)(=O)[OH]", "Polar"),
    "Cl": ("[Cl]", "Halogen"),
    "Br": ("[Br]", "Halogen"),
    "NO2": ("N(=O)=O", "Polar"),
    "CN": ("C#N", "Polar"),
    "C=C": ("C=C", "Hydrophobic"),
    "Aromatic": ("a", "Hydrophobic"),
    "Ether": ("C-O-C", "Polar"),
    "Ester": ("C(=O)-O-C", "Polar"),
}


def get_functional_groups_for_molecule(smiles):
    """Identify functional groups in a molecule"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    fg_dict = {}
    for fg_name, (smarts_pattern, _) in FUNCTIONAL_GROUP_SMARTS.items():
        substructure = Chem.MolFromSmarts(smarts_pattern)
        if substructure is None:
            continue
        matches = mol.GetSubstructMatches(substructure)
        if matches:
            atom_indices = set()
            for match in matches:
                for idx in match:
                    atom_indices.add(idx)
            if atom_indices:
                fg_dict[fg_name] = list(atom_indices)

    return fg_dict


# ==================== Model Loading ====================
def load_pretrained_model(model_path, new_prediction_tasks):
    """Load pretrained model"""
    print(f"Loading transfer-learned model: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    loaded_model_state = checkpoint.get('model_state_dict', checkpoint)

    # Reconstruct model based on MODEL_CONFIG
    original_model_type = MODEL_CONFIG["model_type"]
    input_dim = MODEL_CONFIG["params"]["input_dim"]
    hidden_dim = MODEL_CONFIG["params"]["hidden_dim"]
    num_layers = MODEL_CONFIG["params"]["num_layers"]
    dropout = MODEL_CONFIG["params"]["dropout"]

    model_params = {
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'prediction_tasks': new_prediction_tasks,
        'input_dim': input_dim,
    }

    if original_model_type == "GNN":
        model_params['model_type'] = MODEL_CONFIG["type_specific_params"].get('GNN', {}).get('model_type', 'GCN')
        model = MoleculeGNN(**model_params)
    elif original_model_type == "GAT":
        model_params['num_heads'] = MODEL_CONFIG["type_specific_params"].get('GAT', {}).get('num_heads', 4)
        model = MoleculeGAT(**model_params)
    elif original_model_type == "GIN":
        model = MoleculeGIN(**model_params)
    elif original_model_type == "JKNet":
        model_params['jk_mode'] = MODEL_CONFIG["type_specific_params"].get('JKNet', {}).get('jk_mode', 'cat')
        model = MoleculeJKNet(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {original_model_type}")

    # Load weights (strict=False to allow prediction layer mismatch)
    model.load_state_dict(loaded_model_state, strict=False)
    print("Model weights loaded successfully (strict=False)")

    model.prediction_tasks = new_prediction_tasks
    return model


# ==================== Model Wrapper ====================
class ModelWrapper(nn.Module):
    """Model wrapper for GNNExplainer"""

    def __init__(self, original_model):
        super(ModelWrapper, self).__init__()
        self.model = original_model

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if isinstance(output, tuple):
            prediction = output[0]
        else:
            prediction = output
        return prediction


# ==================== GNNExplainer for Atomic Contributions ====================
def compute_atomic_contributions(model, smiles_list, device, num_epochs=50,
                                 max_molecules=None, verbose=True):
    """Batch compute atomic contributions (using correct MoleculeGraphBuilder)"""
    graph_builder = MoleculeGraphBuilder()
    results = []
    model_wrapped = ModelWrapper(model).to(device)

    # Limit number of molecules if specified
    if max_molecules and len(smiles_list) > max_molecules:
        if verbose:
            print(f"Limiting to {max_molecules} out of {len(smiles_list)} molecules for atomic contributions")
        # Random sampling but ensure diversity
        np.random.seed(42)
        indices = np.random.choice(len(smiles_list), max_molecules, replace=False)
        smiles_list = [smiles_list[i] for i in indices]

    if verbose:
        iterator = tqdm(smiles_list, desc="Computing atomic contributions")
    else:
        iterator = smiles_list

    for i, smiles in enumerate(iterator):
        try:
            # Build molecular graph
            pyg_data = graph_builder.smiles_to_pyg_data(smiles)
            if pyg_data is None:
                if verbose:
                    print(f"Warning: Could not build graph data for {smiles}")
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                if verbose:
                    print(f"Warning: Could not parse SMILES {smiles}")
                continue

            # Prepare data
            pyg_data = pyg_data.to(device)
            if pyg_data.batch is None:
                pyg_data.batch = torch.zeros(pyg_data.num_nodes, dtype=torch.long, device=device)

            # Create explainer
            explainer = Explainer(
                model=model_wrapped,
                algorithm=GNNExplainer(epochs=num_epochs),
                explanation_type="model",
                node_mask_type="attributes",
                edge_mask_type=None,
                model_config=dict(
                    mode="regression",
                    task_level="graph",
                    return_type="raw"
                ),
            )

            # Compute attribution
            explanation = explainer(
                pyg_data.x,
                pyg_data.edge_index,
                batch=pyg_data.batch
            )

            node_mask = explanation.node_mask
            if node_mask.dim() > 1:
                atom_contributions = node_mask.mean(dim=1).cpu().detach().numpy()
            else:
                atom_contributions = node_mask.cpu().detach().numpy()

            # Get atom types
            atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

            results.append({
                'smiles': smiles,
                'contributions': atom_contributions,
                'atom_symbols': atom_symbols,
                'mol': mol,
                'pyg_data': pyg_data
            })

        except Exception as e:
            if verbose:
                print(f"Error processing {smiles}: {e}")
            continue

    return results


# ==================== Improved Visualization Functions ====================
def visualize_atomic_attribution(mol, atom_contributions, output_path,
                                 title="Atomic Contribution Heatmap",
                                 num_id=None, total_CF=None):
    """Visualize atomic contributions as heatmap with improved clarity"""

    # Ensure contributions match number of atoms
    num_atoms = mol.GetNumAtoms()
    if len(atom_contributions) > num_atoms:
        atom_contributions = atom_contributions[:num_atoms]
    elif len(atom_contributions) < num_atoms:
        # Pad with zeros if needed
        atom_contributions = np.pad(atom_contributions,
                                    (0, num_atoms - len(atom_contributions)),
                                    'constant', constant_values=0)

    # Find min and max for normalization
    contrib_min = atom_contributions.min()
    contrib_max = atom_contributions.max()

    # Check if contributions are all positive or all negative
    if contrib_min >= 0:
        # All positive contributions - use sequential colormap
        cmap_name = 'Reds'
        cmap = plt.cm.Reds
        norm = Normalize(vmin=contrib_min, vmax=contrib_max)
        is_diverging = False
    elif contrib_max <= 0:
        # All negative contributions - use sequential colormap (blues)
        cmap_name = 'Blues_r'  # Reverse blues for negative values
        cmap = plt.cm.Blues_r
        norm = Normalize(vmin=contrib_min, vmax=contrib_max)
        is_diverging = False
    else:
        # Both positive and negative - use diverging colormap
        cmap_name = 'coolwarm'
        cmap = plt.cm.coolwarm
        # Use TwoSlopeNorm to center at 0
        vmax = max(abs(contrib_min), abs(contrib_max))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        is_diverging = True

    # Create figure with two subplots: molecule and colorbar
    fig = plt.figure(figsize=(12, 10))

    # Create grid for subplots
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.1)

    # First subplot: molecule
    ax1 = fig.add_subplot(gs[0])

    # Prepare atom colors
    atom_colors = {}
    for i, atom in enumerate(mol.GetAtoms()):
        if i < len(atom_contributions):
            rgba = cmap(norm(atom_contributions[i]))
            atom_colors[atom.GetIdx()] = (rgba[0], rgba[1], rgba[2])

    # Generate 2D coordinates if not present
    try:
        rdDepictor.Compute2DCoords(mol)
    except:
        pass

    # Draw molecule
    drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
    drawer.SetFontSize(0.8)

    # Highlight atoms
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(range(num_atoms)),
        highlightAtomColors=atom_colors
    )
    drawer.FinishDrawing()

    # Convert to image
    img_data = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(img_data))

    # Display image
    ax1.imshow(img)
    ax1.axis('off')

    # Add title with information
    title_text = title
    if total_CF is not None:
        title_text += f"\ntotal_CF = {total_CF}"
    if num_id is not None:
        title_text += f" | num_id = {num_id}"

    # Add contribution statistics
    stats_text = f"Min: {contrib_min:.4f}, Max: {contrib_max:.4f}, Mean: {atom_contributions.mean():.4f}"
    ax1.set_title(title_text, fontsize=25, fontweight='bold', pad=20)
    ax1.text(0.5, 0.02, stats_text, transform=ax1.transAxes,
             ha='center', fontsize=25, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Second subplot: colorbar
    ax2 = fig.add_subplot(gs[1])

    # Create colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add colorbar
    cbar = plt.colorbar(sm, cax=ax2, orientation='vertical')

    # Set colorbar label
    if is_diverging:
        cbar.set_label('Atomic Contribution\n(Blue: Negative, Red: Positive)',
                       fontsize=25, labelpad=15)
    else:
        if contrib_min >= 0:
            cbar.set_label('Positive Atomic Contribution', fontsize=25, labelpad=15)
        else:
            cbar.set_label('Negative Atomic Contribution', fontsize=25, labelpad=15)

    # Adjust colorbar ticks for better readability
    if is_diverging:
        tick_positions = np.linspace(-vmax, vmax, 5)
        tick_labels = [f'{x:.4f}' for x in tick_positions]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)
    else:
        tick_positions = np.linspace(contrib_min, contrib_max, 5)
        tick_labels = [f'{x:.4f}' for x in tick_positions]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)

    # Add text explaining what we're seeing
    explanation_text = "Atomic contributions show how much each atom\ncontributes to the predicted toxicity.\n"
    if is_diverging:
        explanation_text += "Red atoms increase toxicity,\nBlue atoms decrease toxicity."
    else:
        if contrib_min >= 0:
            explanation_text += "All atoms contribute positively to toxicity."
        else:
            explanation_text += "All atoms contribute negatively to toxicity."

    ax2.text(0.5, -0.05, explanation_text, transform=ax2.transAxes,
             ha='center', fontsize=25, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    ax2.tick_params(axis='both', labelsize=20)
    plt.subplots_adjust( top=0.2,
    bottom=0.2,
    left=0.1,
    right=0.95,
    hspace=0.2,
    wspace=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    plt.close()

    print(f"Improved atomic contribution plot saved: {output_path}")

    # Also create a simplified version for quick viewing
    create_simple_atom_contribution_plot(mol, atom_contributions,
                                         output_path.replace('.png', '_simple.png'),
                                         title, num_id, total_CF)

    return fig, (ax1, ax2)
def create_simple_atom_contribution_plot(mol, atom_contributions, output_path,
                                         title="Atomic Contribution",
                                         num_id=None, total_CF=None):
    """Create a simpler version of the atom contribution plot"""
    num_atoms = mol.GetNumAtoms()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 16))

    # Plot 1: Bar chart of atom contributions
    atom_indices = np.arange(num_atoms)
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    # Color bars based on contribution value
    colors = []
    for val in atom_contributions:
        if val < 0:
            # Blue for negative
            colors.append(plt.cm.Blues(0.7))
        elif val > 0:
            # Red for positive
            colors.append(plt.cm.Reds(0.7))
        else:
            # Gray for zero
            colors.append('gray')

    bars = ax1.bar(atom_indices, atom_contributions, color=colors, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add atom symbols as x-tick labels
    ax1.set_xticks(atom_indices)
    ax1.set_xticklabels(atom_symbols, rotation=45, ha='right', fontsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.set_xlabel('Atom Index', fontsize=15)
    ax1.set_ylabel('Contribution Value', fontsize=20)
    ax1.set_title('Atom Contributions', fontsize=20, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, atom_contributions)):
        height = bar.get_height()
        if abs(height) > 0.001:  # Only label significant values
            va = 'bottom' if height >= 0 else 'top'
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{val:.4f}', ha='center', va=va, fontsize=25)

    # Plot 2: Molecular structure with highlights
    # Prepare atom colors for visualization
    contrib_min = atom_contributions.min()
    contrib_max = atom_contributions.max()

    if contrib_min >= 0 or contrib_max <= 0:
        # All same sign - use sequential colormap
        if contrib_min >= 0:
            cmap = plt.cm.Reds
            norm = Normalize(vmin=contrib_min, vmax=contrib_max)
        else:
            cmap = plt.cm.Blues_r
            norm = Normalize(vmin=contrib_min, vmax=contrib_max)
    else:
        # Both signs - use diverging
        cmap = plt.cm.coolwarm
        vmax = max(abs(contrib_min), abs(contrib_max))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    atom_colors = {}
    for i, atom in enumerate(mol.GetAtoms()):
        if i < len(atom_contributions):
            rgba = cmap(norm(atom_contributions[i]))
            atom_colors[atom.GetIdx()] = (rgba[0], rgba[1], rgba[2])

    # Draw molecule
    try:
        rdDepictor.Compute2DCoords(mol)
    except:
        pass

    img = Draw.MolToImage(mol, size=(400, 300),
                          highlightAtoms=list(range(num_atoms)),
                          highlightAtomColors=atom_colors)

    ax2.imshow(img)
    ax2.axis('off')

    # Add title
    plot_title = title
    if total_CF is not None:
        plot_title += f"\ntotal_CF = {total_CF}"
    if num_id is not None:
        plot_title += f" | num_id = {num_id}"
    ax2.set_title(plot_title, fontsize=25, fontweight='bold')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04,
                 label='Contribution Value')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Simple atom contribution plot saved: {output_path}")


# ==================== 修改：修复的实验1函数 ====================
def experiment1_chain_length_dominance_flexible(df, model, device,
                                              selection_mode="single",
                                              selected_indices=None,
                                              max_molecules_per_group=None,  # 不再使用
                                              combination_name=None,
                                              class_stats_df=None):
    """Modified Experiment 1: Compute atomic contributions for ALL valid molecules (no sampling)"""
    print(f"\n{'=' * 60}")
    print(f"Experiment 1: Validating 'Chain Length Dominates Toxicity'")
    print(f"Mode: Process ALL valid molecules (NO sampling)")
    print(f"{'=' * 60}")

    # 1. Get class combinations with statistics if not provided
    if class_stats_df is None:
        class_stats_df = get_class_combinations_with_statistics(df)

    # 2. Select subset based on mode
    if selection_mode == "single":
        if selected_indices is None:
            print("Error: Need to provide selected_indices for single mode")
            return None
        idx = selected_indices[0] if isinstance(selected_indices, list) else selected_indices
        if idx >= len(class_stats_df):
            print(f"Error: Index {idx} out of range (max: {len(class_stats_df) - 1})")
            return None
        target_class = class_stats_df.iloc[idx]['first_class']
        target_subclass = class_stats_df.iloc[idx]['second_class']
        condition = f"(first_class == '{target_class}') & (second_class == '{target_subclass}')"
        subset = df.query(condition).copy()
        subset['source_class'] = f"{target_class}_{target_subclass}"
        if combination_name is None:
            combination_name = f"{target_class}_{target_subclass}"
        print(f"\nSelected: {target_class} - {target_subclass}")
        print(f"Samples: {len(subset)}")

    elif selection_mode == "multiple":
        if not selected_indices:
            print("Error: Need to provide selected_indices for multiple mode")
            return None
        subsets = []
        for idx in selected_indices:
            if 0 <= idx < len(class_stats_df):
                target_class = class_stats_df.iloc[idx]['first_class']
                target_subclass = class_stats_df.iloc[idx]['second_class']
                condition = f"(first_class == '{target_class}') & (second_class == '{target_subclass}')"
                subset_i = df.query(condition).copy()
                subset_i['source_class'] = f"{target_class}_{target_subclass}"
                subsets.append(subset_i)
        if not subsets:
            print("Error: No valid groups selected")
            return None
        subset = pd.concat(subsets, ignore_index=True)
        if combination_name is None:
            combination_name = f"combined_{len(selected_indices)}_groups"
        print(f"\nCombined {len(subsets)} groups. Total samples: {len(subset)}")

    elif selection_mode == "all":
        subset = df.copy()
        print(len(subset))
        subset['source_class'] = subset.apply(
            lambda row: f"{row['first_class']}_{row['second_class']}", axis=1
        )
        if combination_name is None:
            combination_name = "all_molecules"
        print(f"\nUsing ALL molecules. Total samples: {len(subset)}")

    else:
        print(f"Error: Unknown selection mode: {selection_mode}")
        return None

    # 3. Ensure predicted toxicity exists
    if 'predicted_toxicity' not in subset.columns or subset['predicted_toxicity'].isna().all():
        print("Warning: No predicted toxicity values, please run prediction script first")
        return None

    # 4. Filter valid molecules
    valid_indices = subset['predicted_toxicity'].notna()
    subset_valid = subset[valid_indices].copy()
    if len(subset_valid) == 0:
        print("Error: No valid toxicity values")
        return None

    print(f"Processing {len(subset_valid)} molecules for atomic contributions (NO sampling)")

    # 5. Compute atomic contributions for ALL valid molecules
    smiles_list = subset_valid['SMILES'].tolist()
    atomic_data = compute_atomic_contributions(
        model, smiles_list, device,
        num_epochs=NUM_EPOCHS_FOR_EXPLANATION,
        max_molecules=None,  # No limit
        verbose=True
    )
    # with open("20260528.txt", "w") as f:
    #     for data in atomic_data:
    #         mol = data['mol']
    #         smi = data['smiles']
    #         f.write(f"{smi}\n")
    #         contributions = data['contributions']
    #         for atom in mol.GetAtoms():
    #             atom_idx = atom.GetIdx()
    #             if atom_idx < len(contributions):
    #                 atom_contrib = contributions[atom_idx]
    #                 f.write(f"{atom.GetSymbol()} {atom_idx} {atom_contrib}\n")
    # quit()
    if not atomic_data:
        print("Warning: Could not compute atomic contributions")
        cf_contributions = []
        analyzed_smiles = []
    else:
        # Compute CF contributions
        cf_contributions = []
        analyzed_smiles = []
        for data in atomic_data:
            mol = data['mol']
            contributions = data['contributions']
            cf2_contrib = cf3_contrib = 0
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'C':
                    f_count = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == 'F')
                    atom_idx = atom.GetIdx()
                    if atom_idx < len(contributions):
                        atom_contrib = contributions[atom_idx]
                        if f_count == 2:
                            cf2_contrib += atom_contrib
                        elif f_count == 3:
                            cf3_contrib += atom_contrib
            total_cf_contrib = cf2_contrib + cf3_contrib
            cf_contributions.append(total_cf_contrib)
            analyzed_smiles.append(data['smiles'])

        # Merge back
        contribution_df = pd.DataFrame({
            'SMILES': analyzed_smiles,
            'cf_contributions': cf_contributions
        })
        subset_valid = subset_valid.merge(contribution_df, on='SMILES', how='inner')

    # 6. Fit overall regression
    X_all = subset_valid['total_CF'].values.reshape(-1, 1)
    y_all = subset_valid['predicted_toxicity'].values
    reg_all = LinearRegression()
    reg_all.fit(X_all, y_all)
    r_all, p_val = pearsonr(subset_valid['total_CF'], subset_valid['predicted_toxicity'])
    X_sorted = np.sort(X_all.flatten())
    y_pred_sorted = reg_all.predict(X_sorted.reshape(-1, 1))

    # 7. Plotting with improved clarity for large data and no legend clutter
    fig, ax1 = plt.subplots(figsize=(20, 15))  # Larger figure

    # For 'all' mode, use a single color for all points to avoid clutter
    if selection_mode == "all":
        ax1.scatter(subset_valid['total_CF'], subset_valid['predicted_toxicity'],
                    alpha=0.4, color='red', s=100, label='All Molecules',
                    edgecolors='none')  # No edge to reduce clutter
    else:
        # For single/multiple mode, keep group colors but simplify the legend
        if 'source_class' in subset_valid.columns:
            unique_classes = subset_valid['source_class'].unique()
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
            color_dict = {cls: colors[i] for i, cls in enumerate(unique_classes)}
            for cls, color in color_dict.items():
                group_data = subset_valid[subset_valid['source_class'] == cls]
                ax1.scatter(group_data['total_CF'], group_data['predicted_toxicity'],
                            alpha=0.4, color=color, s=60, label=cls,
                            edgecolors='none')  # No edge to reduce clutter
        else:
            ax1.scatter(subset_valid['total_CF'], subset_valid['predicted_toxicity'],
                        alpha=0.5, color='red', s=100, label='All Molecules')

    # Overall trend
    ax1.plot(X_sorted, y_pred_sorted, 'k-', linewidth=2.5, label=f'Chain Length vs oral rat $LD₅₀$ (Pearson $r={r_all:.3f}$)')
    ax1.tick_params(axis='both', labelsize=18)
    # CF contribution scatter (blue triangles)
    if 'cf_contributions' in subset_valid.columns and len(subset_valid) > 1:
        ax2 = ax1.twinx()
        ax2.scatter(subset_valid['total_CF'], subset_valid['cf_contributions'],
                    alpha=0.6, color='blue', s=100, marker='^',
                    edgecolors='none', label='C-F Contribution Sum')

        # Fit CF contribution trend
        X_cf = subset_valid['total_CF'].values.reshape(-1, 1)
        y_cf = subset_valid['cf_contributions'].values
        reg_cf = LinearRegression()
        reg_cf.fit(X_cf, y_cf)
        y_cf_fit = reg_cf.predict(np.sort(X_cf.flatten()).reshape(-1, 1))
        r_cf, _ = pearsonr(subset_valid['total_CF'],
                           subset_valid['cf_contributions'])
        ax2.plot(np.sort(X_cf.flatten()), y_cf_fit, 'b--', linewidth=2,
                 label=f'Chain Length vs C-F Contribution(Pearson $r = {r_cf:.3f}$)')
        ax2.set_ylabel('C-F Atomic Contribution Sum', color='blue', fontsize=20)
        ax2.tick_params(axis='y', labelcolor='blue')
        # Combine legends into a single, clean legend at the bottom
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # Place legend at the bottom, horizontally aligned
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
                   bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=20)
    else:
        # If no CF contribution, just show the main legend
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=20)

    ax1.set_xlabel('Fluorinated Carbon Chain Length ', fontsize=20)
    ax1.set_ylabel(f'$LD₅₀(-log(mol/kg)$', fontsize=20)
    ax1.grid(True, alpha=0.3)

    # Title
    # title_str = f"Chain Length vs Toxicity (All Valid Molecules): {combination_name}\n"
    # title_str += f"Pearson $r = {r_all:.3f}$"
    # if 'cf_contributions' in subset_valid.columns:
    #     r_all, _ = pearsonr(subset_valid['total_CF'], subset_valid['predicted_toxicity'])
    #     title_str += f" | C-F Contribution Pearson $r = {r_all:.3f}$"
    # plt.title(title_str, fontsize=20, fontweight='bold')

    # Adjust layout to make room for the bottom legend
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    safe_name = combination_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    plt.savefig(f'experiment1_all_molecules_{safe_name}.png', dpi=500, bbox_inches='tight')
    plt.show()

    # 9. Report
    print(f"\n{'=' * 60}")
    print("EXPERIMENT 1: ALL-MOLECULE ANALYSIS REPORT")
    print(f"{'=' * 60}")
    print(f"Total valid molecules: {len(subset_valid)}")
    print(f"Overall Pearson r: {r_all:.3f}")

    # 10. Save results
    columns_to_save = ['SMILES', 'first_class', 'second_class', 'total_CF', 'predicted_toxicity']
    if 'source_class' in subset_valid.columns:
        columns_to_save.insert(2, 'source_class')
    if 'cf_contributions' in subset_valid.columns:
        columns_to_save.append('cf_contributions')
    result_df = subset_valid[columns_to_save]
    result_df.to_csv(f'experiment1_all_molecules_results_{safe_name}.csv', index=False)

    summary_data = {
        'selection_mode': selection_mode,
        'combination_name': combination_name,
        'total_molecules': len(subset),
        'valid_molecules': len(subset_valid),
        'analyzed_molecules': len(subset_valid),
        'overall_r': r_all,
        'cf_contribution_r': reg_cf.score(X_cf, y_cf) if 'cf_contributions' in subset_valid.columns else np.nan,
        'hypothesis_support': 'Strong' if r_all >= 0.7 else 'Moderate' if r_all >= 0.5 else 'Weak'
    }
    pd.DataFrame([summary_data]).to_csv(f'experiment1_all_molecules_summary_{safe_name}.csv', index=False)

    print(f"\nResults saved:")
    print(f"  - Plot: experiment1_all_molecules_{safe_name}.png")
    print(f"  - Data: experiment1_all_molecules_results_{safe_name}.csv")
    print(f"  - Summary: experiment1_all_molecules_summary_{safe_name}.csv")

    return {
        'subset': result_df,
        'summary': summary_data,
        'group_results': {},
        'overall_r': r_all
    }


# ==================== 辅助函数 ====================
def get_class_combinations_with_statistics(df, max_display=30):
    """获取带有统计信息的类别组合列表"""
    class_combinations = df.groupby(['first_class', 'second_class']).size().reset_index()
    class_combinations.columns = ['first_class', 'second_class', 'count']
    class_combinations = class_combinations.sort_values(by='count', ascending=False)

    # 计算统计信息
    stats_list = []

    for i, (_, row) in enumerate(class_combinations.iterrows()):
        target_class = row['first_class']
        target_subclass = row['second_class']

        subset = df[(df['first_class'] == target_class) &
                    (df['second_class'] == target_subclass)]

        if len(subset) > 0:
            min_cf = subset['total_CF'].min()
            max_cf = subset['total_CF'].max()
            mean_cf = subset['total_CF'].mean()
            std_cf = subset['total_CF'].std()
            median_cf = subset['total_CF'].median()
            cf_diversity = subset['total_CF'].nunique()
            cf_range = max_cf - min_cf

            # 评估适用性
            if cf_diversity >= 5 and cf_range >= 2:
                suitability = "✓ Good"
                suitability_desc = "Diverse chain lengths, ideal for analysis"
            elif cf_diversity >= 3:
                suitability = "⚠ Moderate"
                suitability_desc = "Moderate chain length variation"
            else:
                suitability = "✗ Limited"
                suitability_desc = "Limited chain length variation"

            stats_list.append({
                'first_class': target_class,
                'second_class': target_subclass,
                'count': row['count'],
                'min_cf': min_cf,
                'max_cf': max_cf,
                'mean_cf': mean_cf,
                'std_cf': std_cf,
                'median_cf': median_cf,
                'cf_diversity': cf_diversity,
                'cf_range': cf_range,
                'suitability': suitability,
                'suitability_desc': suitability_desc
            })

    return pd.DataFrame(stats_list)


def display_class_selection_with_statistics(class_stats_df, start_idx=0, end_idx=None):
    """显示类别选择统计信息"""
    if end_idx is None:
        end_idx = len(class_stats_df)

    print("\n" + "=" * 80)
    print(f"Available class combinations (showing {start_idx + 1}-{end_idx} of {len(class_stats_df)}):")
    print("=" * 80)

    for i in range(start_idx, min(end_idx, len(class_stats_df))):
        row = class_stats_df.iloc[i]

        # 格式化显示
        first_class = row['first_class'][:25].ljust(25)
        second_class = row['second_class'][:30].ljust(30)

        print(f"{i + 1:3d}. {first_class} - {second_class}")
        print(f"     Samples: {row['count']:3d} | Chain Length: {row['min_cf']:.1f}-{row['max_cf']:.1f} "
              f"(mean={row['mean_cf']:.1f}±{row['std_cf']:.1f})")
        print(f"     Median: {row['median_cf']:.1f} | Distinct lengths: {row['cf_diversity']} | "
              f"Suitability: {row['suitability']}")
        print(f"     Description: {row['suitability_desc']}")
        print()

    print("=" * 80)
    print("Selection Guide:")
    print("  ✓ Good: Diverse chain lengths (≥5 distinct values, range ≥2) - Ideal for analysis")
    print("  ⚠ Moderate: Moderate chain length variation (≥3 distinct values)")
    print("  ✗ Limited: Limited chain length variation - Not ideal for chain length analysis")
    print("=" * 80)


def interactive_class_selection(class_stats_df, page_size=15):
    """交互式类别选择，支持分页和多选"""
    total_pages = (len(class_stats_df) + page_size - 1) // page_size
    current_page = 0

    while True:
        print(f"\n{'=' * 80}")
        print(f"Page {current_page + 1} of {total_pages}")
        print("=" * 80)

        start_idx = current_page * page_size
        end_idx = min((current_page + 1) * page_size, len(class_stats_df))

        display_class_selection_with_statistics(class_stats_df, start_idx, end_idx)

        print("\nNavigation and Selection:")
        print("  Enter numbers to select (comma-separated for multiple, e.g., '1,3,5')")
        print("  'n' for next page, 'p' for previous page")
        print("  'a' to select all Good suitability groups")
        print("  'all' to use ALL molecules")
        print("  'q' to finish selection and proceed")
        print("  'c' to clear selection")
        print("  's' to skip to experiment setup")

        user_input = input("\nYour choice: ").strip().lower()

        if user_input == 'q':
            break
        elif user_input == 'n' and current_page < total_pages - 1:
            current_page += 1
        elif user_input == 'p' and current_page > 0:
            current_page -= 1
        elif user_input == 'a':
            # Select all Good suitability groups
            good_indices = class_stats_df[class_stats_df['suitability'] == '✓ Good'].index.tolist()
            return good_indices, "multiple"
        elif user_input == 'all':
            return None, "all"
        elif user_input == 'c':
            return [], "multiple"
        elif user_input == 's':
            return None, "skip"
        elif user_input:
            try:
                # Parse multiple selections
                selections = [int(x.strip()) - 1 for x in user_input.split(',')]
                # Convert to actual indices
                actual_indices = [start_idx + s for s in selections
                                  if 0 <= s < (end_idx - start_idx)]

                if actual_indices:
                    return actual_indices, "multiple" if len(actual_indices) > 1 else "single"
                else:
                    print("Invalid selection, please try again.")
            except ValueError:
                print("Invalid input, please try again.")

    return [], "multiple"


# ==================== 保留原有的实验二 ====================
def experiment2_functional_group_fixed_effect(df, model, device, target_groups=None, max_samples_per_group=20):
    """Experiment 2: Identify fixed effects of functional groups"""
    print(f"\n{'=' * 60}")
    print(f"Experiment 2: Identifying Functional Group 'Fixed Effects'")
    print(f"{'=' * 60}")

    if target_groups is None:
        target_groups = ['I', 'OH', 'SO3H', 'COOH', 'CF3']

    # 1. Collect molecules containing target functional groups
    all_relevant_smiles = []

    # For each functional group, find ALL molecules containing it
    for fg in target_groups:
        # Find molecules containing this functional group
        fg_molecules = []
        for idx, row in df.iterrows():
            smiles = row['SMILES']
            fg_dict = get_functional_groups_for_molecule(smiles)
            if fg in fg_dict:
                fg_molecules.append(smiles)

        print(f"Functional group {fg}: Found {len(fg_molecules)} molecules")

        # Limit to reasonable number if too many
        if len(fg_molecules) > max_samples_per_group:
            # Select diverse molecules
            selected = fg_molecules[:max_samples_per_group]
            print(f"  -> Selecting {len(selected)} molecules for analysis")
        else:
            selected = fg_molecules

        all_relevant_smiles.extend(selected)

    # Remove duplicates
    all_relevant_smiles = list(set(all_relevant_smiles))
    print(f"\nTotal unique molecules for analysis: {len(all_relevant_smiles)}")

    if len(all_relevant_smiles) == 0:
        print("Error: No molecules found with target functional groups")
        return None

    # 2. Calculate atomic contributions for ALL relevant molecules
    print(f"Calculating atomic contributions for {len(all_relevant_smiles)} molecules...")
    atomic_data = compute_atomic_contributions(model, all_relevant_smiles, device,
                                               num_epochs=NUM_EPOCHS_FOR_EXPLANATION)

    if not atomic_data:
        print("Warning: Could not compute atomic contributions")
        return None

    # 3. Collect functional group contribution values
    fg_contributions = {fg: [] for fg in target_groups}

    for data in atomic_data:
        mol = data['mol']
        contributions = data['contributions']
        smiles = data['smiles']

        # Get functional groups for this molecule
        fg_dict = get_functional_groups_for_molecule(Chem.MolToSmiles(mol))

        # Calculate mean contribution for each functional group
        for fg_name in target_groups:
            if fg_name in fg_dict:
                atom_indices = fg_dict[fg_name]
                valid_indices = [i for i in atom_indices if i < len(contributions)]
                if valid_indices:
                    mean_contrib = np.mean([contributions[i] for i in valid_indices])
                    fg_contributions[fg_name].append(mean_contrib)

    # 4. Plot violin plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: All functional group distribution comparison
    ax1 = axes[0]
    data_to_plot = []
    labels = []

    for fg in target_groups:
        if len(fg_contributions[fg]) > 0:
            data_to_plot.append(fg_contributions[fg])
            labels.append(fg)

    if data_to_plot:
        violin_parts = ax1.violinplot(data_to_plot, showmeans=True, showmedians=True)

        # Beautify violin plot
        for pc in violin_parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        # Add median markers
        ax1.scatter(range(1, len(labels) + 1),
                    [np.median(data) for data in data_to_plot],
                    color='red', marker='o', s=200, label='Median', zorder=3)

        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Zero Baseline')
        ax1.set_xticks(range(1, len(labels) + 1))
        ax1.set_xticklabels(labels, rotation=45, fontsize=20)
        ax1.tick_params(axis='x', labelsize=20)
        ax1.set_ylabel('Atomic Contribution Value', fontsize=20)
        ax1.set_title('Contribution Value Distribution by Functional Group', fontsize=40, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
    else:
        ax1.text(0.5, 0.5, 'No target functional groups found',
                 ha='center', va='center', transform=ax1.transAxes, fontsize=40)
        ax1.set_title('Contribution Value Distribution by Functional Group', fontsize=40, fontweight='bold')

    # Right: Detailed distribution for specific functional group
    ax2 = axes[1]
    if len(fg_contributions.get('I', [])) > 0:
        # Box plot
        bp = ax2.boxplot(fg_contributions['I'], vert=True, patch_artist=True,
                         boxprops=dict(facecolor='lightgreen'),
                         medianprops=dict(color='red', linewidth=2),
                         whiskerprops=dict(color='black', linewidth=1.5),
                         capprops=dict(color='black', linewidth=1.5),
                         flierprops=dict(marker='o', markersize=200
                                         , alpha=0.5))

        # Add data points
        x_jitter = np.random.normal(1, 0.05, len(fg_contributions['I']))
        ax2.scatter(x_jitter, fg_contributions['I'], alpha=0.6, color='blue', s=50)

        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xticks([1])
        ax2.set_xticklabels(['I (Halogen)'], fontsize=20)
        ax2.tick_params(axis='x', labelsize=20)
        ax2.set_ylabel('Contribution Value', fontsize=40)

        # Statistical information
        mean_val = np.mean(fg_contributions['I'])
        median_val = np.median(fg_contributions['I'])
        std_val = np.std(fg_contributions['I'])

        stats_text = f"Sample size: {len(fg_contributions['I'])}\n"
        stats_text += f"Mean: {mean_val:.3f}\n"
        stats_text += f"Median: {median_val:.3f}\n"
        stats_text += f"Std Dev: {std_val:.3f}"

        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                 verticalalignment='top', fontsize=40,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax2.set_title('I Functional Group Fixed Effect Analysis', fontsize=40, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'No I functional groups found',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=40)
        ax2.set_title('I Functional Group Fixed Effect Analysis', fontsize=40, fontweight='bold')

    plt.suptitle("Experiment 2: Functional Group Fixed Effects Validation", fontsize=40, fontweight='bold')
    plt.tight_layout()
    plt.savefig('experiment2_functional_group_fixed_effect.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 5. Save results
    result_data = []
    for fg, contribs in fg_contributions.items():
        if contribs:
            result_data.append({
                'functional_group': fg,
                'mean_contribution': np.mean(contribs),
                'median_contribution': np.median(contribs),
                'std_contribution': np.std(contribs),
                'sample_size': len(contribs)
            })

    if result_data:
        result_df = pd.DataFrame(result_data)
        result_df.to_csv('experiment2_results.csv', index=False)

        print(f"\nExperiment 2 completed:")
        print(f"  - Analyzed functional groups: {target_groups}")
        print(f"  - Functional groups with data: {[fg for fg in target_groups if fg_contributions[fg]]}")
        print(f"  - Plot saved: experiment2_functional_group_fixed_effect.png")
        print(f"  - Data saved: experiment2_results.csv")
    else:
        print("Warning: No contribution data found for any target functional groups")

    return fg_contributions


# ==================== 保留原有的实验三 ====================
def experiment3_branching_reduces_toxicity(df, model, device, target_class="PFCAs",
                                           max_molecules_per_type=50, visualize_pairs=3):
    """
    Experiment 3: Validate 'Branched Chains Reduce Toxicity' (Matching Fluorinated Carbon Count)

    Parameters:
    - df: DataFrame containing molecular data
    - model: Trained GNN model
    - device: Computing device
    - target_class: Target compound class
    - max_molecules_per_type: Maximum number of molecules per type
    - visualize_pairs: Number of molecule pairs to visualize

    Returns:
    - Results DataFrame
    - List of matched pairs
    """

    print(f"\n{'=' * 80}")
    print(f"Experiment 3: Validate 'Branched Chains Reduce Toxicity' (Matching Fluorinated Carbon Count)")
    print(f"Target Class: {target_class}")
    print(f"{'=' * 80}")

    # 1. Filter straight and branched chain molecules
    straight_molecules = df[(df['chain_type'] == 'straight') &
                            (df['first_class'] == target_class)].copy()
    branched_molecules = df[(df['chain_type'] == 'branched') &
                            (df['first_class'] == target_class)].copy()

    print(f"Straight chain molecules: {len(straight_molecules)}")
    print(f"Branched chain molecules: {len(branched_molecules)}")

    if len(straight_molecules) == 0 or len(branched_molecules) == 0:
        print("Error: No straight or branched chain molecules found")
        return None, None

    # 2. Analyze fluorinated carbon count distribution
    print(f"\nFluorinated Carbon Count Analysis:")

    # Get all possible CF values
    straight_cf_values = sorted(straight_molecules['total_CF'].unique())
    branched_cf_values = sorted(branched_molecules['total_CF'].unique())

    print(f"Straight chain CF values: {straight_cf_values}")
    print(f"Branched chain CF values: {branched_cf_values}")

    # Find common CF values
    common_cf_values = sorted(set(straight_cf_values) & set(branched_cf_values))

    if common_cf_values:
        print(f"\nCommon CF values: {common_cf_values}")

        # Select CF value with most samples
        cf_sample_counts = []
        for cf in common_cf_values:
            straight_count = len(straight_molecules[straight_molecules['total_CF'] == cf])
            branched_count = len(branched_molecules[branched_molecules['total_CF'] == cf])
            total_count = straight_count + branched_count
            cf_sample_counts.append((cf, total_count, straight_count, branched_count))

        # Sort by sample count
        cf_sample_counts.sort(key=lambda x: x[1], reverse=True)

        print(f"\nCommon CF value sample distribution:")
        for cf, total, s_count, b_count in cf_sample_counts:
            print(f"  CF={cf}: Total {total} (Straight {s_count}, Branched {b_count})")
    else:
        print(f"\nWarning: No common CF values")
        print(f"Will find closest CF value pairs")

        # Find closest CF value pairs
        closest_pairs = []
        for scf in straight_cf_values:
            for bcf in branched_cf_values:
                diff = abs(scf - bcf)
                straight_count = len(straight_molecules[straight_molecules['total_CF'] == scf])
                branched_count = len(branched_molecules[branched_molecules['total_CF'] == bcf])
                closest_pairs.append((scf, bcf, diff, straight_count, branched_count))

        # Sort by difference, then by total sample count
        closest_pairs.sort(key=lambda x: (x[2], -(x[3] + x[4])))

        if closest_pairs:
            # Select best pairs
            selected_pairs = closest_pairs[:min(5, len(closest_pairs))]
            print(f"\nClosest CF value pairs:")
            for scf, bcf, diff, s_count, b_count in selected_pairs:
                print(f"  Straight CF={scf} ({s_count}) <-> Branched CF={bcf} ({b_count}), Difference={diff}")

    # 3. Find matched molecule pairs
    matched_pairs = []

    if common_cf_values:
        # Use common CF values for matching
        for cf in common_cf_values:
            straight_at_cf = straight_molecules[straight_molecules['total_CF'] == cf]
            branched_at_cf = branched_molecules[branched_molecules['total_CF'] == cf]

            # Create all possible pairs
            for _, s_row in straight_at_cf.iterrows():
                for _, b_row in branched_at_cf.iterrows():
                    matched_pairs.append({
                        'straight_smiles': s_row['SMILES'],
                        'branched_smiles': b_row['SMILES'],
                        'straight_toxicity': s_row.get('predicted_toxicity', np.nan),
                        'branched_toxicity': b_row.get('predicted_toxicity', np.nan),
                        'cf_value': cf,
                        'toxicity_diff': b_row.get('predicted_toxicity', np.nan) - s_row.get('predicted_toxicity',
                                                                                             np.nan)
                    })
    else:
        # Use closest CF values for matching
        for scf, bcf, diff, s_count, b_count in selected_pairs:
            straight_at_scf = straight_molecules[straight_molecules['total_CF'] == scf]
            branched_at_bcf = branched_molecules[branched_molecules['total_CF'] == bcf]

            # Limit number of pairs
            s_sample = straight_at_scf.head(min(5, len(straight_at_scf)))
            b_sample = branched_at_bcf.head(min(5, len(branched_at_bcf)))

            for _, s_row in s_sample.iterrows():
                for _, b_row in b_sample.iterrows():
                    matched_pairs.append({
                        'straight_smiles': s_row['SMILES'],
                        'branched_smiles': b_row['SMILES'],
                        'straight_toxicity': s_row.get('predicted_toxicity', np.nan),
                        'branched_toxicity': b_row.get('predicted_toxicity', np.nan),
                        'straight_cf': scf,
                        'branched_cf': bcf,
                        'cf_diff': diff,
                        'toxicity_diff': b_row.get('predicted_toxicity', np.nan) - s_row.get('predicted_toxicity',
                                                                                             np.nan)
                    })

    print(f"\nFound {len(matched_pairs)} matched molecule pairs")

    if len(matched_pairs) == 0:
        print("Error: No matched molecule pairs found")
        return None, None

    # 4. Sort by toxicity difference, select pairs for visualization
    matched_pairs.sort(key=lambda x: abs(x['toxicity_diff']), reverse=True)
    pairs_to_visualize = min(visualize_pairs, len(matched_pairs))

    print(f"\nSelecting top {pairs_to_visualize} molecule pairs for visualization analysis:")

    # 5. Collect all SMILES for atomic contribution calculation
    all_smiles = []
    pair_details = []

    for i, pair in enumerate(matched_pairs[:pairs_to_visualize]):
        all_smiles.append(pair['straight_smiles'])
        all_smiles.append(pair['branched_smiles'])
        pair_details.append((i, pair))

        cf_info = f"CF={pair['cf_value']}" if 'cf_value' in pair else f"Straight CF={pair['straight_cf']}, Branched CF={pair['branched_cf']}"
        print(f"  Pair{i + 1}: {cf_info}, Toxicity difference={pair['toxicity_diff']:.3f}")

    # 6. Calculate atomic contributions
    print(f"\nCalculating atomic contributions for {len(all_smiles)} molecules...")

    # Use existing compute_atomic_contributions function
    atomic_data = compute_atomic_contributions(
        model, all_smiles, device,
        num_epochs=NUM_EPOCHS_FOR_EXPLANATION,
        max_molecules=None,
        verbose=True
    )

    if not atomic_data:
        print("Error: Could not compute atomic contributions")
        return None, None

    # 7. Organize atomic contribution data into dictionary for quick access
    atom_contrib_dict = {}
    for data in atomic_data:
        smiles = data['smiles']
        atom_contrib_dict[smiles] = {
            'mol': data['mol'],
            'contributions': data['contributions'],
            'atom_symbols': data['atom_symbols']
        }

    # 8. Generate heatmaps and statistical analysis for each pair
    pair_results = []

    for pair_idx, pair in pair_details:
        straight_smiles = pair['straight_smiles']
        branched_smiles = pair['branched_smiles']

        if straight_smiles not in atom_contrib_dict or branched_smiles not in atom_contrib_dict:
            print(f"Warning: Molecules in pair{pair_idx + 1} lack atomic contribution data")
            continue

        # Get molecular data
        straight_data = atom_contrib_dict[straight_smiles]
        branched_data = atom_contrib_dict[branched_smiles]

        # Calculate statistics
        straight_contrib = straight_data['contributions']
        branched_contrib = branched_data['contributions']

        straight_stats = {
            'mean': np.mean(straight_contrib),
            'std': np.std(straight_contrib),
            'max': np.max(straight_contrib),
            'min': np.min(straight_contrib),
            'positive_count': np.sum(straight_contrib > 0),
            'negative_count': np.sum(straight_contrib < 0),
            'total_atoms': len(straight_contrib)
        }

        branched_stats = {
            'mean': np.mean(branched_contrib),
            'std': np.std(branched_contrib),
            'max': np.max(branched_contrib),
            'min': np.min(branched_contrib),
            'positive_count': np.sum(branched_contrib > 0),
            'negative_count': np.sum(branched_contrib < 0),
            'total_atoms': len(branched_contrib)
        }

        # Store pair results
        cf_info = {
            'straight_cf': pair['cf_value'] if 'cf_value' in pair else pair['straight_cf'],
            'branched_cf': pair['cf_value'] if 'cf_value' in pair else pair['branched_cf']
        }

        pair_results.append({
            'pair_id': pair_idx + 1,
            'straight_smiles': straight_smiles,
            'branched_smiles': branched_smiles,
            'straight_toxicity': pair['straight_toxicity'],
            'branched_toxicity': pair['branched_toxicity'],
            'toxicity_diff': pair['toxicity_diff'],
            'cf_info': cf_info,
            'cf_diff': pair.get('cf_diff', 0),
            'straight_stats': straight_stats,
            'branched_stats': branched_stats
        })

        # 9. Generate heatmaps for each pair
        print(f"\nGenerating heatmaps for pair{pair_idx + 1}...")

        # Straight molecule heatmap
        straight_title = f"Straight Chain (Pair{pair_idx + 1})\nToxicity: {pair['straight_toxicity']:.3f}"
        if 'cf_value' in pair:
            straight_title += f" | CF={pair['cf_value']}"
        else:
            straight_title += f" | CF={pair['straight_cf']}"

        straight_output = f"experiment3_pair{pair_idx + 1}_straight_tox{pair['straight_toxicity']:.3f}.png"

        visualize_atomic_attribution(
            mol=straight_data['mol'],
            atom_contributions=straight_contrib,
            output_path=straight_output,
            title=straight_title,
            total_CF=cf_info['straight_cf']
        )

        # Branched molecule heatmap
        branched_title = f"Branched Chain (Pair{pair_idx + 1})\nToxicity: {pair['branched_toxicity']:.3f}"
        if 'cf_value' in pair:
            branched_title += f" | CF={pair['cf_value']}"
        else:
            branched_title += f" | CF={pair['branched_cf']}"

        branched_output = f"experiment3_pair{pair_idx + 1}_branched_tox{pair['branched_toxicity']:.3f}.png"

        visualize_atomic_attribution(
            mol=branched_data['mol'],
            atom_contributions=branched_contrib,
            output_path=branched_output,
            title=branched_title,
            total_CF=cf_info['branched_cf']
        )

    # 10. Generate overall statistical analysis
    print(f"\n{'=' * 80}")
    print(f"Overall Statistical Analysis")
    print(f"{'=' * 80}")

    if not pair_results:
        print("No pair results to analyze")
        return None, None

    # Create results DataFrame
    results_data = []
    for result in pair_results:
        # Extract fluorinated carbon information
        if result['cf_diff'] == 0:
            cf_match = "✓ Perfect match"
            cf_value = result['cf_info']['straight_cf']
        else:
            cf_match = f"⚠ Approximate match (Difference: {result['cf_diff']})"
            cf_value = f"{result['cf_info']['straight_cf']} vs {result['cf_info']['branched_cf']}"

        # Determine hypothesis validation results
        if result['toxicity_diff'] < 0:
            toxicity_support = "✓ Supported (Branched less toxic)"
        else:
            toxicity_support = "✗ Not supported (Branched equally or more toxic)"

        if result['branched_stats']['std'] > result['straight_stats']['std']:
            distribution_support = "✓ Supported (Branched contributions more dispersed)"
        else:
            distribution_support = "✗ Not supported (Branched contributions less dispersed)"

        results_data.append({
            'Pair_ID': result['pair_id'],
            'CF_Match': cf_match,
            'CF_Value': cf_value,
            'Straight_Toxicity': f"{result['straight_toxicity']:.3f}",
            'Branched_Toxicity': f"{result['branched_toxicity']:.3f}",
            'Toxicity_Difference': f"{result['toxicity_diff']:.3f}",
            'Toxicity_Hypothesis': toxicity_support,
            'Straight_Contribution_Std': f"{result['straight_stats']['std']:.6f}",
            'Branched_Contribution_Std': f"{result['branched_stats']['std']:.6f}",
            'Distribution_Hypothesis': distribution_support,
            'Straight_SMILES': result['straight_smiles'][:30] + "..." if len(result['straight_smiles']) > 30 else
            result['straight_smiles'],
            'Branched_SMILES': result['branched_smiles'][:30] + "..." if len(result['branched_smiles']) > 30 else
            result['branched_smiles']
        })

    results_df = pd.DataFrame(results_data)

    # 11. Plot overall comparison charts
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Subplot 1: Toxicity value comparison (all pairs)
    ax1 = axes[0, 0]
    pair_ids = [f"Pair{r['pair_id']}" for r in pair_results]
    straight_toxicities = [r['straight_toxicity'] for r in pair_results]
    branched_toxicities = [r['branched_toxicity'] for r in pair_results]

    x = np.arange(len(pair_ids))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, straight_toxicities, width, label='Straight', color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width / 2, branched_toxicities, width, label='Branched', color='orange', alpha=0.7)

    ax1.set_xlabel('Molecule Pair', fontsize=16)
    ax1.set_ylabel('Predicted Toxicity (pLD₅₀)', fontsize=16)
    ax1.set_title('Straight vs Branched Toxicity Comparison (by Pair)', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pair_ids, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Subplot 2: Toxicity difference distribution
    ax2 = axes[0, 1]
    toxicity_diffs = [r['toxicity_diff'] for r in pair_results]

    colors = ['green' if diff < 0 else 'red' for diff in toxicity_diffs]
    bars = ax2.bar(pair_ids, toxicity_diffs, color=colors, alpha=0.7)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Molecule Pair', fontsize=16)
    ax2.set_ylabel('Toxicity Difference (Branched - Straight)', fontsize=16)
    ax2.set_title('Toxicity Difference Distribution', fontsize=16, fontweight='bold')
    ax2.set_xticks(range(len(pair_ids)))
    ax2.set_xticklabels(pair_ids, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, diff in zip(bars, toxicity_diffs):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{diff:.3f}', ha='center', va=va, fontsize=16)

    # Subplot 3: Contribution standard deviation comparison
    ax3 = axes[0, 2]
    straight_stds = [r['straight_stats']['std'] for r in pair_results]
    branched_stds = [r['branched_stats']['std'] for r in pair_results]

    bars3 = ax3.bar(x - width / 2, straight_stds, width, label='Straight', color='lightblue', alpha=0.7)
    bars4 = ax3.bar(x + width / 2, branched_stds, width, label='Branched', color='lightcoral', alpha=0.7)

    ax3.set_xlabel('Molecule Pair', fontsize=16)
    ax3.set_ylabel('Contribution Standard Deviation', fontsize=16)
    ax3.set_title('Contribution Distribution Dispersion Comparison', fontsize=16, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(pair_ids, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Subplot 4: Contribution distribution of first pair (straight)
    ax4 = axes[1, 0]
    if len(pair_results) > 0:
        first_pair = pair_results[0]
        straight_contrib = atom_contrib_dict[first_pair['straight_smiles']]['contributions']

        ax4.hist(straight_contrib, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Baseline')
        ax4.set_xlabel('Atomic Contribution Value', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title(f'Straight Chain Contribution Distribution (Pair 1)', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Add statistical information
        stats_text = f"Toxicity: {first_pair['straight_toxicity']:.3f}\n"
        stats_text += f"Mean: {first_pair['straight_stats']['mean']:.6f}\n"
        stats_text += f"Std: {first_pair['straight_stats']['std']:.6f}\n"
        stats_text += f"Atoms: {first_pair['straight_stats']['total_atoms']}"

        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
                 verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Subplot 5: Contribution distribution of first pair (branched)
    ax5 = axes[1, 1]
    if len(pair_results) > 0:
        branched_contrib = atom_contrib_dict[first_pair['branched_smiles']]['contributions']

        ax5.hist(branched_contrib, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax5.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Baseline')
        ax5.set_xlabel('Atomic Contribution Value', fontsize=12)
        ax5.set_ylabel('Frequency', fontsize=12)
        ax5.set_title(f'Branched Chain Contribution Distribution (Pair 1)', fontsize=14)
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Add statistical information
        stats_text = f"Toxicity: {first_pair['branched_toxicity']:.3f}\n"
        stats_text += f"Mean: {first_pair['branched_stats']['mean']:.6f}\n"
        stats_text += f"Std: {first_pair['branched_stats']['std']:.6f}\n"
        stats_text += f"Atoms: {first_pair['branched_stats']['total_atoms']}"

        ax5.text(0.02, 0.98, stats_text, transform=ax5.transAxes,
                 verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Subplot 6: Hypothesis validation summary
    ax6 = axes[1, 2]
    ax6.axis('off')

    # Calculate support statistics
    toxicity_support_count = sum(1 for r in pair_results if r['toxicity_diff'] < 0)
    distribution_support_count = sum(1 for r in pair_results if r['branched_stats']['std'] > r['straight_stats']['std'])

    total_pairs = len(pair_results)
    toxicity_support_rate = toxicity_support_count / total_pairs * 100
    distribution_support_rate = distribution_support_count / total_pairs * 100

    summary_text = f"Experiment 3: Branched Chains Reduce Toxicity Validation\n"
    summary_text += f"Target Class: {target_class}\n"
    summary_text += f"Pairs Analyzed: {total_pairs}\n\n"

    summary_text += f"Hypothesis 1: Branched chains are less toxic\n"
    summary_text += f"  Supported: {toxicity_support_count}/{total_pairs} ({toxicity_support_rate:.1f}%)\n"
    if toxicity_support_rate > 50:
        summary_text += f"  ✓ Overall supports this hypothesis\n"
    else:
        summary_text += f"  ✗ Overall does not support this hypothesis\n"

    summary_text += f"\nHypothesis 2: Branched chain contributions are more dispersed\n"
    summary_text += f"  Supported: {distribution_support_count}/{total_pairs} ({distribution_support_rate:.1f}%)\n"
    if distribution_support_rate > 50:
        summary_text += f"  ✓ Overall supports this hypothesis\n"
    else:
        summary_text += f"  ✗ Overall does not support this hypothesis\n"

    # Add CF matching information
    perfect_matches = sum(1 for r in pair_results if r['cf_diff'] == 0)
    if perfect_matches > 0:
        summary_text += f"\nCF Matching Status:\n"
        summary_text += f"  Perfect matches: {perfect_matches}/{total_pairs}\n"
        summary_text += f"  Approximate matches: {total_pairs - perfect_matches}/{total_pairs}\n"

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             verticalalignment='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.suptitle(f"Experiment 3: Branched vs Straight Chain Comparison (Matching CF Count) - {target_class}",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'experiment3_matched_CF_{target_class}_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 12. Save results
    results_df.to_csv(f'experiment3_matched_CF_{target_class}_results.csv', index=False)

    print(f"\n{'=' * 80}")
    print(f"Experiment Completed!")
    print(f"Results Saved:")
    print(f"  1. Summary plot: experiment3_matched_CF_{target_class}_summary.png")
    print(f"  2. Results table: experiment3_matched_CF_{target_class}_results.csv")
    print(f"  3. Atomic heatmaps: experiment3_pair*_*.png (Total {len(pair_results) * 2} images)")
    print(f"{'=' * 80}")

    return results_df, pair_results


# ==================== 主程序 ====================
def main():
    """Enhanced main program with bug fixes and original experiments"""
    print("=" * 80)
    print("GNN Explanation Experiments: Complete SAR Analysis Suite")
    print("=" * 80)

    # 1. Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Load model
    print("\nLoading model...")
    model = load_pretrained_model(PRETRAINED_MODEL_PATH, new_prediction_tasks=1)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 3. Load prediction results
    print("\nLoading prediction results...")
    if os.path.exists(PREDICTIONS_CACHE_PATH):
        predictions_df = pd.read_csv(PREDICTIONS_CACHE_PATH)
        print(f"Loaded predictions from cache: {len(predictions_df)} records")

        df = df_enhanced.merge(
            predictions_df[['SMILES', 'predicted_toxicity']],
            on='SMILES',
            how='left'
        )
    else:
        df = df_enhanced.copy()
        df['predicted_toxicity'] = np.nan
        print("Warning: No prediction cache found, toxicity values will be NaN")

    # Check toxicity values
    valid_samples = df['predicted_toxicity'].notna().sum()
    print(f"Valid toxicity prediction samples: {valid_samples}/{len(df)}")

    if valid_samples == 0:
        print("Error: All toxicity values are NaN, please run prediction script first")
        return

    # 4. Get class statistics
    class_stats_df = get_class_combinations_with_statistics(df)

    # 5. Enhanced experiment selection menu
    while True:
        print("\n" + "=" * 80)
        print("MAIN EXPERIMENT SELECTION MENU")
        print("=" * 80)
        print("1. Experiment 1: Validate 'Chain Length Dominates Toxicity'")
        print("   a. Single class analysis")
        print("   b. Multiple classes combined")
        print("   c. All molecules together")
        print("2. Experiment 2: Identify Functional Group 'Fixed Effects'")
        print("3. Experiment 3: Support 'Branched Chains Reduce Toxicity'")
        print("4. Run All Experiments (with default settings)")
        print("5. Custom Experiment Combination")
        print("6. Exit")
        print("=" * 80)

        choice = input("\nEnter your choice (1-6): ").strip()

        if choice == '1':
            # Experiment 1 submenu
            print("\n" + "=" * 80)
            print("EXPERIMENT 1: Chain Length Dominates Toxicity")
            print("=" * 80)
            print("Select analysis mode:")
            print("a. Single class analysis")
            print("b. Multiple classes combined")
            print("c. All molecules together")
            print("d. Back to main menu")

            sub_choice = input("\nEnter choice (a-d): ").strip().lower()

            if sub_choice == 'a':
                # Single class analysis
                print("\n>>> Single Class Analysis <<<")
                selected_indices, mode = interactive_class_selection(class_stats_df)

                if mode == "single" and selected_indices:
                    experiment1_chain_length_dominance_flexible(
                        df=df,
                        model=model,
                        device=device,
                        selection_mode="single",
                        selected_indices=selected_indices,
                        max_molecules_per_group=50,
                        class_stats_df=class_stats_df
                    )
                elif mode == "skip":
                    continue

            elif sub_choice == 'b':
                # Multiple classes combined
                print("\n>>> Multiple Classes Combined Analysis <<<")
                selected_indices, mode = interactive_class_selection(class_stats_df)

                if mode == "multiple" and selected_indices:
                    # Get names for combination
                    selected_names = []
                    for idx in selected_indices:
                        if 0 <= idx < len(class_stats_df):
                            row = class_stats_df.iloc[idx]
                            selected_names.append(f"{row['first_class']}_{row['second_class']}")

                    combination_name = f"combined_{len(selected_names)}_groups"

                    experiment1_chain_length_dominance_flexible(
                        df=df,
                        model=model,
                        device=device,
                        selection_mode="multiple",
                        selected_indices=selected_indices,
                        max_molecules_per_group=50,
                        combination_name=combination_name,
                        class_stats_df=class_stats_df
                    )
                elif mode == "skip":
                    continue

            elif sub_choice == 'c':
                # All molecules together
                print("\n>>> All Molecules Analysis <<<")

                experiment1_chain_length_dominance_flexible(
                    df=df,
                    model=model,
                    device=device,
                    selection_mode="all",
                    selected_indices=None,
                    max_molecules_per_group=100,
                    combination_name="all_molecules_comprehensive",
                    class_stats_df=class_stats_df
                )

            elif sub_choice == 'd':
                continue

        elif choice == '2':
            # Experiment 2
            print("\n" + "=" * 80)
            print("EXPERIMENT 2: Functional Group Fixed Effects")
            print("=" * 80)

            # Ask for target groups
            default_groups = ['I', 'OH', 'SO3H', 'COOH', 'CF3']
            print(f"\nDefault functional groups: {default_groups}")
            custom_groups = input("Enter custom groups (comma-separated, or press Enter for default): ").strip()

            if custom_groups:
                target_groups = [g.strip() for g in custom_groups.split(',')]
            else:
                target_groups = default_groups

            experiment2_functional_group_fixed_effect(
                df=df,
                model=model,
                device=device,
                target_groups=target_groups,
                max_samples_per_group=20
            )

        elif choice == '3':
            # Experiment 3
            print("\n" + "=" * 80)
            print("EXPERIMENT 3: Branched Chains Reduce Toxicity")
            print("=" * 80)

            # Show available classes
            print("\nAvailable classes with chain_type information:")
            available_classes = df[df['chain_type'].notna()]['first_class'].unique()
            for i, cls in enumerate(available_classes[:15]):
                print(f"{i + 1}. {cls}")

            target_class = input("\nEnter target class (or press Enter for default 'PFCAs'): ").strip()
            if not target_class:
                target_class = "PFCAs"

            experiment3_branching_reduces_toxicity(
                df=df,
                model=model,
                device=device,
                target_class=target_class,
                max_molecules_per_type=100
            )

        elif choice == '4':
            # Run all experiments with defaults
            print("\n" + "=" * 80)
            print("RUNNING ALL EXPERIMENTS WITH DEFAULT SETTINGS")
            print("=" * 80)

            # Experiment 1 with default
            print("\n>>> Running Experiment 1 (Default: PFAA precursors - PolyFACs) <<<")
            exp1_result = experiment1_chain_length_dominance_flexible(
                df=df,
                model=model,
                device=device,
                selection_mode="single",
                selected_indices=[0],  # Assuming first one is PFAA precursors - PolyFACs
                max_molecules_per_group=50,
                combination_name="PFAA_precursors_PolyFACs",
                class_stats_df=class_stats_df
            )

            # Experiment 2 with defaults
            print("\n>>> Running Experiment 2 (Default groups) <<<")
            experiment2_functional_group_fixed_effect(
                df=df,
                model=model,
                device=device,
                target_groups=['I', 'OH', 'SO3H', 'COOH', 'CF3'],
                max_samples_per_group=20
            )

            # Experiment 3 with default
            print("\n>>> Running Experiment 3 (Default: PFCAs) <<<")
            experiment3_branching_reduces_toxicity(
                df=df,
                model=model,
                device=device,
                target_class="PFCAs",
                max_molecules_per_type=100
            )

        elif choice == '5':
            # Custom combination
            print("\n" + "=" * 80)
            print("CUSTOM EXPERIMENT COMBINATION")
            print("=" * 80)

            experiments = input("\nEnter experiments to run (comma-separated, e.g., '1a,2,3'): ").strip().lower()

            if '1' in experiments:
                # Determine which version of Experiment 1
                if '1a' in experiments:
                    # Single class
                    selected_indices, mode = interactive_class_selection(class_stats_df)
                    if mode == "single" and selected_indices:
                        experiment1_chain_length_dominance_flexible(
                            df=df,
                            model=model,
                            device=device,
                            selection_mode="single",
                            selected_indices=selected_indices,
                            max_molecules_per_group=50,
                            class_stats_df=class_stats_df
                        )
                elif '1b' in experiments:
                    # Multiple classes
                    selected_indices, mode = interactive_class_selection(class_stats_df)
                    if mode == "multiple" and selected_indices:
                        experiment1_chain_length_dominance_flexible(
                            df=df,
                            model=model,
                            device=device,
                            selection_mode="multiple",
                            selected_indices=selected_indices,
                            max_molecules_per_group=50,
                            class_stats_df=class_stats_df
                        )
                elif '1c' in experiments:
                    # All molecules
                    experiment1_chain_length_dominance_flexible(
                        df=df,
                        model=model,
                        device=device,
                        selection_mode="all",
                        selected_indices=None,
                        max_molecules_per_group=100,
                        combination_name="all_molecules",
                        class_stats_df=class_stats_df
                    )

            if '2' in experiments:
                experiment2_functional_group_fixed_effect(
                    df=df,
                    model=model,
                    device=device,
                    target_groups=['I', 'OH', 'SO3H', 'COOH', 'CF3'],
                    max_samples_per_group=20
                )

            if '3' in experiments:
                experiment3_branching_reduces_toxicity(
                    df=df,
                    model=model,
                    device=device,
                    target_class="PFCAs",
                    max_molecules_per_type=100
                )

        elif choice == '6':
            print("\nExiting program. Goodbye!")
            break

        else:
            print("Invalid choice, please try again.")

    print("\n" + "=" * 80)
    print("All experiments completed!")
    print("Results saved to current directory")
    print("=" * 80)


# ==================== Program Entry Point ====================
if __name__ == "__main__":
    main()