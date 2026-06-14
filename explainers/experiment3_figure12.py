#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced PFAS Branched vs Straight Chain Toxicity Analysis
- Multi-dimensional statistical analysis
- Confounding factor control
- Advanced branch position inference
- Interactive visualization
- Comprehensive reporting
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')

# --- Add project root to path ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Standard imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re
from collections import defaultdict
import textwrap

# --- Interactive visualization imports ---
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Install with: pip install plotly")

# --- GNN & RDKit imports ---
import torch
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdmolops, rdDistGeom, rdDepictor
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem import Descriptors

# --- Statistical modeling imports ---
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols, mixedlm

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels not available. Install with: pip install statsmodels")

# --- Scikit-learn for multivariate analysis ---
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# --- Project module imports ---
from config import get_model_save_path
from dataset.graph_builder import MoleculeGraphBuilder
from models.gnn_model import MoleculeGNN

# ==================== CONFIGURATION ====================
DATA_PATH = r'F:\GNN-pro\data_process_class\step3_OECD_Class_Enhanced_v2_with_CF.csv'
PREDICTIONS_CACHE_PATH = r'F:\GNN-pro\data_process_class\pfas_sar_analysis_predictions.csv'
PRETRAINED_MODEL_PATH = get_model_save_path(
    r"F:\GNN-pro\scripts\outputs\GNN\GNN_transfer_spilts1_500_251107_124134\transfer_learned_best_model.pth"
)

MODEL_CONFIG = {
    "model_type": "GNN",
    "params": {
        "hidden_dim": 256,
        "num_layers": 8,
        "dropout": 0.2,
        "prediction_tasks": 1,
        "input_dim": 120,
    },
    "type_specific_params": {"GNN": {"model_type": "GCN"}}
}

TARGET_MEAN = 2.8219
TARGET_STD = 1.0778
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- SPECIFIC COMBINATIONS TO ANALYZE ---
TARGET_COMBINATIONS = [
    ("PFAA precursors", "HFCs"),
    ("PFAA precursors", "PFAIs"),
    ("PFAA precursors", "PFAenes"),
    ("PFAA precursors", "PolyFACs"),
    ("PFAA precursors", "SFAenes"),
    ("PFAAs", "PFCA-ester derivatives"),
    ("Polyfluoroalkyl acids", "PolyFCAs"),
]

# --- Visualization settings ---
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-whitegrid')

# 全局图例样式设置
plt.rcParams.update({
    'legend.fontsize': 11,
    'legend.title_fontsize': 12,
    'legend.frameon': True,
    'legend.framealpha': 0.85,
    'legend.fancybox': True,
    'legend.edgecolor': 'gray',
    'legend.borderpad': 0.5,
    'legend.labelspacing': 0.5,
    'legend.handlelength': 1.8,
    'legend.handleheight': 0.7,
    'legend.handletextpad': 0.5,
    'legend.borderaxespad': 0.5,
    'legend.columnspacing': 1.5
})


# ==================== ADVANCED BRANCH POSITION INFERENCE ====================
def identify_main_chain_atoms(mol):
    """
    Identify atoms belonging to the main carbon chain in a molecule.
    Uses a heuristic approach to find the longest linear carbon chain.
    """
    if mol is None:
        return []

    # Get all carbon atoms
    carbon_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
    if not carbon_atoms:
        return []

    # Build adjacency list for carbon atoms only
    carbon_indices = [atom.GetIdx() for atom in carbon_atoms]
    adj_list = {idx: [] for idx in carbon_indices}

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in carbon_indices and end_idx in carbon_indices:
            adj_list[begin_idx].append(end_idx)
            adj_list[end_idx].append(begin_idx)

    # Function to perform BFS and find longest path from a starting node
    def bfs_longest_path(start):
        visited = set()
        queue = [(start, [start])]
        longest_path = []

        while queue:
            node, path = queue.pop(0)
            if len(path) > len(longest_path):
                longest_path = path

            visited.add(node)
            for neighbor in adj_list.get(node, []):
                if neighbor not in visited and neighbor not in path:
                    queue.append((neighbor, path + [neighbor]))

        return longest_path

    # Try multiple starting points to find the longest chain
    longest_chain = []
    for start in carbon_indices[:min(10, len(carbon_indices))]:  # Limit to first 10 to save time
        chain = bfs_longest_path(start)
        if len(chain) > len(longest_chain):
            longest_chain = chain

    return longest_chain


def improved_branch_position_inference(smiles):
    """
    Improved branch position inference with multiple heuristics.
    Returns: 'α', 'β', 'γ', 'δ', 'ε', 'ω', 'middle', or 'N/A'
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "N/A"

        # Method 1: SMILES pattern matching
        patterns = [
            (r'C\([^)]*\)C', 'branch_near_start'),  # Branch early in chain
            (r'C[^C]\([^)]*\)', 'branch_near_start'),
            (r'CC\([^)]*\)C', 'branch_early'),
            (r'CCC\([^)]*\)C', 'branch_middle'),
            (r'CCCC\([^)]*\)C', 'branch_late'),
        ]

        for pattern, position_type in patterns:
            if re.search(pattern, smiles):
                if position_type == 'branch_near_start':
                    return "α"
                elif position_type == 'branch_early':
                    return "β"
                elif position_type == 'branch_middle':
                    return "middle"
                elif position_type == 'branch_late':
                    return "ω"

        # Method 2: Molecular topology analysis
        main_chain = identify_main_chain_atoms(mol)
        if not main_chain:
            return "N/A"

        # Find branching points (carbons with degree >= 3)
        branch_points = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:  # Carbon
                neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 6]
                if len(neighbors) >= 3 and atom.GetIdx() in main_chain:
                    branch_points.append(atom.GetIdx())

        if not branch_points:
            return "N/A"

        # Calculate position relative to chain ends
        distances_from_start = []
        for bp in branch_points:
            try:
                # Find position in main chain
                if bp in main_chain:
                    pos = main_chain.index(bp)
                    distances_from_start.append(pos)
            except ValueError:
                continue

        if not distances_from_start:
            return "N/A"

        # Normalize position (0 = start, 1 = end)
        normalized_positions = [d / max(1, len(main_chain) - 1) for d in distances_from_start]
        avg_position = np.mean(normalized_positions)

        # Classify based on normalized position
        if avg_position < 0.1:
            return "α"
        elif avg_position < 0.25:
            return "β"
        elif avg_position < 0.4:
            return "γ"
        elif avg_position < 0.6:
            return "middle"
        elif avg_position < 0.75:
            return "δ"
        elif avg_position < 0.9:
            return "ε"
        else:
            return "ω"

    except Exception as e:
        print(f"Error in branch position inference for {smiles}: {e}")
        return "N/A"


def infer_branch_length_and_complexity(smiles):
    """
    Infer additional branch characteristics.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"length": 0, "complexity": 0, "type": "none"}

        # Count branches (non-main chain carbons)
        main_chain = identify_main_chain_atoms(mol)
        all_carbons = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]

        branch_carbons = [c for c in all_carbons if c not in main_chain]
        branch_length = len(branch_carbons)

        # Calculate branch complexity (average branching in branches)
        complexity = 0
        for carbon in branch_carbons:
            atom = mol.GetAtomWithIdx(carbon)
            carbon_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 6]
            complexity += len(carbon_neighbors)

        complexity = complexity / max(1, branch_length)

        # Classify branch type
        if branch_length == 0:
            btype = "none"
        elif branch_length == 1:
            btype = "methyl"
        elif branch_length == 2:
            btype = "ethyl"
        elif branch_length <= 4:
            btype = "short"
        else:
            btype = "long"

        return {
            "length": branch_length,
            "complexity": complexity,
            "type": btype
        }

    except Exception as e:
        return {"length": 0, "complexity": 0, "type": "error"}


# ==================== SAMPLE BALANCE ANALYSIS ====================
def analyze_sample_balance(straight_df, branched_df, target_combinations):
    """
    Comprehensive sample balance analysis.
    """
    print("\n" + "=" * 60)
    print("SAMPLE BALANCE ANALYSIS")
    print("=" * 60)

    # Basic counts
    print(f"\n1. Overall Sample Counts:")
    print(f"   Straight chains: {len(straight_df)}")
    print(f"   Branched chains: {len(branched_df)}")
    print(f"   Total: {len(straight_df) + len(branched_df)}")
    print(f"   Ratio (Straight:Branched): {len(straight_df)}:{len(branched_df)}")

    # Check minimum sample requirements
    min_samples_required = 10
    if len(straight_df) < min_samples_required or len(branched_df) < min_samples_required:
        print(f"\n⚠️ WARNING: Sample size below recommended minimum ({min_samples_required})")
        print(f"   Statistical power may be limited")

    # Distribution across target combinations
    print(f"\n2. Distribution Across Target Combinations:")

    balance_report = []
    for fc, sc in target_combinations:
        straight_count = len(straight_df[(straight_df['first_class'] == fc) &
                                         (straight_df['second_class'] == sc)])
        branched_count = len(branched_df[(branched_df['first_class'] == fc) &
                                         (branched_df['second_class'] == sc)])

        if straight_count > 0 or branched_count > 0:
            ratio = straight_count / max(1, branched_count) if branched_count > 0 else float('inf')
            imbalance = abs(straight_count - branched_count) / max(straight_count, branched_count)

            balance_report.append({
                "Combination": f"{fc} - {sc}",
                "Straight": straight_count,
                "Branched": branched_count,
                "Ratio": f"{ratio:.2f}:1",
                "Imbalance": f"{imbalance:.1%}"
            })

    # Create balance report dataframe
    if balance_report:
        balance_df = pd.DataFrame(balance_report)
        print(balance_df.to_string(index=False))

        # Identify imbalanced combinations
        imbalanced = balance_df[balance_df['Imbalance'].str.rstrip('%').astype(float) > 50]
        if not imbalanced.empty:
            print(f"\n⚠️ IMBALANCED COMBINATIONS (>50% difference):")
            for _, row in imbalanced.iterrows():
                print(
                    f"   {row['Combination']}: {row['Straight']} vs {row['Branched']} ({row['Imbalance']} difference)")

    # Additional balance metrics
    print(f"\n3. Additional Balance Metrics:")

    # CF chain length distribution
    if 'total_CF' in straight_df.columns and 'total_CF' in branched_df.columns:
        cf_balance = stats.mannwhitneyu(
            straight_df['total_CF'].dropna(),
            branched_df['total_CF'].dropna()
        )
        print(f"   CF chain length distribution similarity (Mann-Whitney U): p = {cf_balance.pvalue:.4f}")

    # Molecular weight distribution
    if 'MW' in straight_df.columns and 'MW' in branched_df.columns:
        mw_balance = stats.mannwhitneyu(
            straight_df['MW'].dropna(),
            branched_df['MW'].dropna()
        )
        print(f"   Molecular weight distribution similarity (Mann-Whitney U): p = {mw_balance.pvalue:.4f}")

    return balance_df if balance_report else None


# ==================== CONFOUNDING FACTOR ANALYSIS ====================
def analyze_confounding_factors(straight_df, branched_df):
    """
    Analyze and control for confounding factors.
    """
    print("\n" + "=" * 60)
    print("CONFOUNDING FACTOR ANALYSIS")
    print("=" * 60)

    confounding_factors = [
        ('total_CF', 'Total Fluorinated Carbons'),
        ('MW', 'Molecular Weight'),
        ('LogP', 'Hydrophobicity (LogP)'),
        ('TPSA', 'Topological Polar Surface Area'),
        ('HBD', 'Hydrogen Bond Donors'),
        ('HBA', 'Hydrogen Bond Acceptors'),
        ('RotatableBonds', 'Rotatable Bonds'),
    ]

    results = []

    for factor, description in confounding_factors:
        if factor in straight_df.columns and factor in branched_df.columns:
            # Remove NaN values
            straight_vals = straight_df[factor].dropna()
            branched_vals = branched_df[factor].dropna()

            if len(straight_vals) > 0 and len(branched_vals) > 0:
                # Basic statistics
                straight_mean = straight_vals.mean()
                branched_mean = branched_vals.mean()
                straight_std = straight_vals.std()
                branched_std = branched_vals.std()

                # Statistical test
                try:
                    # Try t-test for normal distributions
                    _, p_normal_straight = stats.shapiro(straight_vals)
                    _, p_normal_branched = stats.shapiro(branched_vals)

                    if p_normal_straight > 0.05 and p_normal_branched > 0.05:
                        # Normal distribution, use t-test
                        _, p_value = stats.ttest_ind(straight_vals, branched_vals)
                        test_name = "t-test"
                    else:
                        # Non-normal distribution, use Mann-Whitney U
                        _, p_value = stats.mannwhitneyu(straight_vals, branched_vals)
                        test_name = "Mann-Whitney U"
                except:
                    # Fallback to Mann-Whitney U
                    _, p_value = stats.mannwhitneyu(straight_vals, branched_vals)
                    test_name = "Mann-Whitney U"

                # Effect size (Cohen's d for continuous variables)
                n1, n2 = len(straight_vals), len(branched_vals)
                pooled_std = np.sqrt(((n1 - 1) * straight_vals.var() + (n2 - 1) * branched_vals.var()) / (n1 + n2 - 2))
                cohen_d = (straight_mean - branched_mean) / pooled_std if pooled_std != 0 else 0

                # Store results
                results.append({
                    'Factor': description,
                    'Straight Mean': f"{straight_mean:.2f} ± {straight_std:.2f}",
                    'Branched Mean': f"{branched_mean:.2f} ± {branched_std:.2f}",
                    'Difference': f"{straight_mean - branched_mean:.2f}",
                    'P-Value': f"{p_value:.4f}",
                    'Test': test_name,
                    'Cohens_d': f"{cohen_d:.3f}",  # 修改这里：移除单引号
                    'Significant': p_value < 0.05
                })

    # Create results dataframe
    if results:
        results_df = pd.DataFrame(results)
        print("\nConfounding Factor Differences Between Groups:")
        print(results_df.to_string(index=False))

        # Identify significant confounding factors
        significant_confounders = results_df[results_df['Significant']]
        if not significant_confounders.empty:
            print(f"\n⚠️ SIGNIFICANT CONFOUNDING FACTORS (p < 0.05):")
            for _, row in significant_confounders.iterrows():
                # 使用不同的方式避免f-string中的反斜杠
                factor = row['Factor']
                p_val = row['P-Value']
                cohens_d_val = row['Cohens_d']
                print(f"{factor}: p = {p_val}, Cohen's d = {cohens_d_val}")
                print(f"Straight: {row['Straight Mean']}, Branched: {row['Branched Mean']}")

            print(f"\n💡 RECOMMENDATION: Consider using propensity score matching or")
            print(f"   multivariate regression to control for these confounders.")
        else:
            print(f"\n✅ No significant confounding factors detected.")

        # Visualize confounding factors
        visualize_confounding_factors(straight_df, branched_df, confounding_factors)

        return results_df
    else:
        print("No confounding factor data available.")
        return None


def visualize_confounding_factors(straight_df, branched_df, confounding_factors):
    """
    Create visualization for confounding factor comparison.
    """
    available_factors = [(f, d) for f, d in confounding_factors
                         if f in straight_df.columns and f in branched_df.columns]

    if not available_factors:
        return

    # Create subplots
    n_factors = len(available_factors)
    n_cols = min(3, n_factors)
    n_rows = (n_factors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # Flatten axes if necessary
    if n_factors == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (factor, description) in enumerate(available_factors):
        ax = axes[idx]

        # Prepare data
        data = []
        labels = []

        straight_data = straight_df[factor].dropna()
        branched_data = branched_df[factor].dropna()

        if len(straight_data) > 0:
            data.append(straight_data)
            labels.append('Straight')

        if len(branched_data) > 0:
            data.append(branched_data)
            labels.append('Branched')

        if data:
            # Create box plot
            bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)

            # Set colors
            colors = ['skyblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add jittered scatter points
            for i, (group_data, color) in enumerate(zip(data, colors[:len(data)])):
                x_pos = np.random.normal(i + 1, 0.04, size=len(group_data))
                ax.scatter(x_pos, group_data, alpha=0.4, color=color, s=20, edgecolors='black', linewidth=0.5)

            ax.set_title(f"{description}", fontsize=12, fontweight='bold')
            ax.set_ylabel("Value", fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')

            # Add statistical significance if applicable
            if len(data) == 2:
                _, p_value = stats.mannwhitneyu(data[0], data[1])

                if p_value < 0.001:
                    sig_symbol = '***'
                elif p_value < 0.01:
                    sig_symbol = '**'
                elif p_value < 0.05:
                    sig_symbol = '*'
                else:
                    sig_symbol = 'ns'

                # Add significance bracket
                y_max = max(data[0].max(), data[1].max())
                y_min = min(data[0].min(), data[1].min())
                y_range = y_max - y_min

                ax.plot([1, 1, 2, 2],
                        [y_max + 0.1 * y_range, y_max + 0.15 * y_range,
                         y_max + 0.15 * y_range, y_max + 0.1 * y_range],
                        lw=1.5, c='black')
                ax.text(1.5, y_max + 0.2 * y_range, sig_symbol,
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Hide empty subplots
    for idx in range(len(available_factors), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Confounding Factor Comparison Between Straight and Branched Chains",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("confounding_factor_comparison.png", dpi=500, bbox_inches='tight')
    plt.show()


# ==================== MULTIVARIATE REGRESSION ANALYSIS ====================
def perform_multivariate_analysis(combined_df):
    """
    Perform comprehensive multivariate regression analysis.
    """
    print("\n" + "=" * 60)
    print("MULTIVARIATE REGRESSION ANALYSIS")
    print("=" * 60)

    # Prepare data
    combined_df = combined_df.copy()

    # Create binary variables
    combined_df['is_branched'] = (combined_df['chain_type'] == 'branched').astype(int)

    # Encode branch position if available
    if 'branch_position' in combined_df.columns:
        position_mapping = {'α': 1, 'β': 2, 'γ': 3, 'δ': 4, 'ε': 5, 'ω': 6, 'middle': 3.5, 'N/A': 0}
        combined_df['branch_position_encoded'] = combined_df['branch_position'].map(position_mapping).fillna(0)

    # Add branch characteristics if available
    if 'SMILES' in combined_df.columns:
        print("Inferring branch characteristics...")
        branch_info = combined_df['SMILES'].apply(infer_branch_length_and_complexity)
        combined_df['branch_length'] = branch_info.apply(lambda x: x['length'])
        combined_df['branch_complexity'] = branch_info.apply(lambda x: x['complexity'])
        combined_df['branch_type'] = branch_info.apply(lambda x: x['type'])

    # Define potential features
    base_features = ['total_CF', 'MW', 'LogP', 'TPSA', 'is_branched']
    optional_features = ['branch_position_encoded', 'branch_length', 'branch_complexity',
                         'HBD', 'HBA', 'RotatableBonds']

    # Check which features are available
    available_features = [f for f in base_features if f in combined_df.columns]
    available_features += [f for f in optional_features if f in combined_df.columns]

    # Check if target variable exists
    if 'predicted_toxicity' not in combined_df.columns:
        print("Warning: 'predicted_toxicity' not found in dataset.")
        return None

    # Remove rows with missing values
    analysis_df = combined_df[available_features + ['predicted_toxicity']].dropna()

    if len(analysis_df) < 20:
        print(f"Warning: Insufficient data for multivariate analysis (n={len(analysis_df)})")
        return None

    print(f"Multivariate analysis with {len(analysis_df)} samples and {len(available_features)} features")

    # 1. Linear Regression
    print("\n1. LINEAR REGRESSION ANALYSIS")

    X = analysis_df[available_features]
    y = analysis_df['predicted_toxicity']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LinearRegression()
    lr.fit(X_scaled, y)
    y_pred = lr.predict(X_scaled)

    r2_lr = r2_score(y, y_pred)
    mse_lr = mean_squared_error(y, y_pred)

    print(f"   R² Score: {r2_lr:.4f}")
    print(f"   MSE: {mse_lr:.4f}")

    # Feature coefficients
    print("\n   Feature Coefficients (standardized):")
    coef_df = pd.DataFrame({
        'Feature': available_features,
        'Coefficient': lr.coef_,
        'Abs_Coefficient': np.abs(lr.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)

    print(coef_df.to_string(index=False))

    # 2. Regularized Regression (LASSO)
    print("\n2. LASSO REGRESSION (Feature Selection)")

    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(X_scaled, y)

    lasso_coef_df = pd.DataFrame({
        'Feature': available_features,
        'LASSO_Coefficient': lasso.coef_,
        'Selected': np.abs(lasso.coef_) > 0.001
    }).sort_values('LASSO_Coefficient', ascending=False)

    print(lasso_coef_df.to_string(index=False))

    selected_features = lasso_coef_df[lasso_coef_df['Selected']]['Feature'].tolist()
    print(f"\n   Features selected by LASSO: {selected_features}")

    # 3. Random Forest
    print("\n3. RANDOM FOREST REGRESSION")

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)

    # Cross-validation
    cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='r2')
    print(f"   Cross-validated R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # Feature importance
    rf_importance_df = pd.DataFrame({
        'Feature': available_features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n   Feature Importance (Random Forest):")
    print(rf_importance_df.to_string(index=False))

    # Permutation importance
    print("\n   Permutation Importance:")
    perm_importance = permutation_importance(rf, X_scaled, y, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({
        'Feature': available_features,
        'Permutation_Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Permutation_Importance', ascending=False)

    print(perm_df.to_string(index=False))

    # 4. Interaction Effects
    print("\n4. INTERACTION EFFECT ANALYSIS")

    if STATSMODELS_AVAILABLE and len(selected_features) > 0:
        try:
            # Create interaction terms
            interaction_terms = []
            for i, feat1 in enumerate(selected_features):
                for feat2 in selected_features[i + 1:]:
                    if feat1 != 'is_branched' and feat2 != 'is_branched':
                        continue  # Focus on interactions with chain type

                    interaction_col = f"{feat1}:{feat2}"
                    analysis_df[interaction_col] = analysis_df[feat1] * analysis_df[feat2]
                    interaction_terms.append(interaction_col)

            if interaction_terms:
                # Fit model with interactions
                formula = f"predicted_toxicity ~ {' + '.join(selected_features)} + {' + '.join(interaction_terms)}"
                model = ols(formula, data=analysis_df).fit()

                print("   Significant Interaction Effects (p < 0.05):")
                interactions = model.pvalues[model.pvalues.index.str.contains(':')]
                significant_interactions = interactions[interactions < 0.05]

                if len(significant_interactions) > 0:
                    for interaction, pval in significant_interactions.items():
                        print(f"     {interaction}: p = {pval:.4f}")
                else:
                    print("     No significant interaction effects found.")
        except Exception as e:
            print(f"   Interaction analysis failed: {e}")

    # Visualize feature importance
    visualize_feature_importance(rf_importance_df, perm_df)

    return {
        'linear_regression': {
            'coefficients': coef_df,
            'r2': r2_lr,
            'mse': mse_lr
        },
        'lasso': {
            'selected_features': selected_features,
            'coefficients': lasso_coef_df
        },
        'random_forest': {
            'importance': rf_importance_df,
            'permutation_importance': perm_df,
            'cv_scores': cv_scores
        }
    }


def visualize_feature_importance(rf_importance_df, perm_df):
    """
    Visualize feature importance from different models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Random Forest Feature Importance
    ax1 = axes[0]
    rf_sorted = rf_importance_df.sort_values('Importance', ascending=True)
    y_pos = np.arange(len(rf_sorted))

    ax1.barh(y_pos, rf_sorted['Importance'], color='steelblue', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(rf_sorted['Feature'])
    ax1.set_xlabel('Feature Importance', fontsize=12)
    ax1.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.tick_params(axis='both', labelsize=10)

    # Add values on bars
    for i, v in enumerate(rf_sorted['Importance']):
        ax1.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)

    # Permutation Importance
    ax2 = axes[1]
    perm_sorted = perm_df.sort_values('Permutation_Importance', ascending=True)
    y_pos = np.arange(len(perm_sorted))

    ax2.barh(y_pos, perm_sorted['Permutation_Importance'],
             xerr=perm_sorted['Std'],
             color='darkorange', alpha=0.7, ecolor='black', capsize=5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(perm_sorted['Feature'])
    ax2.set_xlabel('Permutation Importance', fontsize=12)
    ax2.set_title('Permutation Importance (± std)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.tick_params(axis='both', labelsize=10)

    plt.suptitle("Feature Importance Analysis", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("feature_importance_analysis.png", dpi=500, bbox_inches='tight')
    plt.show()


# ==================== ADVANCED STATISTICAL METHODS ====================
def perform_advanced_statistical_analysis(straight_tox, branched_tox):
    """
    Perform advanced statistical analysis including bootstrap and Bayesian methods.
    """
    print("\n" + "=" * 60)
    print("ADVANCED STATISTICAL ANALYSIS")
    print("=" * 60)

    # Remove NaN values
    straight_tox = straight_tox[~np.isnan(straight_tox)]
    branched_tox = branched_tox[~np.isnan(branched_tox)]

    if len(straight_tox) < 10 or len(branched_tox) < 10:
        print("Warning: Insufficient data for advanced analysis")
        return None

    results = {}

    # 1. Bootstrap Confidence Intervals
    print("\n1. BOOTSTRAP CONFIDENCE INTERVALS")

    def mean_difference(data1, data2):
        return np.mean(data1) - np.mean(data2)

    # Perform bootstrap
    n_bootstrap = 1000
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample1 = np.random.choice(straight_tox, size=len(straight_tox), replace=True)
        sample2 = np.random.choice(branched_tox, size=len(branched_tox), replace=True)
        bootstrap_diffs.append(mean_difference(sample1, sample2))

    # Calculate confidence intervals
    ci_95 = np.percentile(bootstrap_diffs, [2.5, 97.5])
    ci_90 = np.percentile(bootstrap_diffs, [5, 95])

    print(f"   Mean difference: {np.mean(straight_tox) - np.mean(branched_tox):.3f}")
    print(f"   95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
    print(f"   90% CI: [{ci_90[0]:.3f}, {ci_90[1]:.3f}]")

    results['bootstrap'] = {
        'mean_difference': np.mean(straight_tox) - np.mean(branched_tox),
        'ci_95': ci_95,
        'ci_90': ci_90,
        'bootstrap_diffs': bootstrap_diffs
    }

    # 2. Bayesian Estimation
    print("\n2. BAYESIAN ESTIMATION (Approximate)")

    # Use normal approximation for Bayesian estimation
    straight_mean = np.mean(straight_tox)
    branched_mean = np.mean(branched_tox)
    straight_se = np.std(straight_tox) / np.sqrt(len(straight_tox))
    branched_se = np.std(branched_tox) / np.sqrt(len(branched_tox))

    # Prior: weakly informative (mean=0, sd=2)
    prior_mean = 0
    prior_sd = 2

    # Likelihood
    diff_mean = straight_mean - branched_mean
    diff_se = np.sqrt(straight_se ** 2 + branched_se ** 2)

    # Posterior (conjugate normal-normal)
    post_precision = 1 / prior_sd ** 2 + 1 / diff_se ** 2
    post_mean = (prior_mean / prior_sd ** 2 + diff_mean / diff_se ** 2) / post_precision
    post_sd = np.sqrt(1 / post_precision)

    # Credible intervals
    cred_95 = [post_mean - 1.96 * post_sd, post_mean + 1.96 * post_sd]
    cred_90 = [post_mean - 1.645 * post_sd, post_mean + 1.645 * post_sd]

    print(f"   Posterior mean difference: {post_mean:.3f}")
    print(f"   Posterior standard deviation: {post_sd:.3f}")
    print(f"   95% Credible Interval: [{cred_95[0]:.3f}, {cred_95[1]:.3f}]")
    print(f"   90% Credible Interval: [{cred_90[0]:.3f}, {cred_90[1]:.3f}]")

    # Probability that difference > 0
    prob_positive = 1 - stats.norm.cdf(0, loc=post_mean, scale=post_sd)
    print(f"   Probability that difference > 0: {prob_positive:.1%}")

    results['bayesian'] = {
        'posterior_mean': post_mean,
        'posterior_sd': post_sd,
        'credible_95': cred_95,
        'credible_90': cred_90,
        'prob_positive': prob_positive
    }

    # 3. Power Analysis
    print("\n3. STATISTICAL POWER ANALYSIS")

    # Calculate effect size (Cohen's d)
    n1, n2 = len(straight_tox), len(branched_tox)
    pooled_std = np.sqrt(((n1 - 1) * np.var(straight_tox) + (n2 - 1) * np.var(branched_tox)) / (n1 + n2 - 2))
    cohen_d = (np.mean(straight_tox) - np.mean(branched_tox)) / pooled_std

    # Calculate achieved power
    # Using approximation for two-sample t-test
    from scipy.stats import t
    df = n1 + n2 - 2
    alpha = 0.05
    ncp = cohen_d * np.sqrt(n1 * n2 / (n1 + n2))  # Non-centrality parameter

    critical_t = t.ppf(1 - alpha / 2, df)
    power = 1 - t.cdf(critical_t, df, ncp) + t.cdf(-critical_t, df, ncp)

    print(f"   Effect size (Cohen's d): {cohen_d:.3f}")
    print(f"   Achieved power (α=0.05): {power:.1%}")
    print(f"   Total sample size: {n1 + n2}")

    # Required sample size for 80% power
    if cohen_d > 0:
        from statsmodels.stats.power import TTestIndPower
        power_analysis = TTestIndPower()
        required_n = power_analysis.solve_power(effect_size=cohen_d, power=0.8, alpha=0.05, ratio=n2 / n1)
        print(f"   Required sample size per group for 80% power: {required_n:.0f}")

    results['power_analysis'] = {
        'cohen_d': cohen_d,
        'achieved_power': power,
        'sample_size_straight': n1,
        'sample_size_branched': n2,
        'total_sample_size': n1 + n2
    }

    # Visualize bootstrap results
    visualize_advanced_statistics(results, straight_tox, branched_tox)

    return results


def optimize_legend(ax, fontsize=11, loc='best', ncol=1, title=None):
    """
    统一优化图例样式
    """
    legend = ax.legend(fontsize=fontsize,
                       loc=loc,
                       ncol=ncol,
                       title=title,
                       frameon=True,
                       fancybox=True,
                       framealpha=0.9,
                       edgecolor='black',
                       title_fontsize=fontsize + 1)

    # 调整图例位置
    if loc == 'upper right':
        legend.set_bbox_to_anchor((0.98, 0.98))

    return legend


def visualize_advanced_statistics(results, straight_tox, branched_tox):
    """
    Visualize advanced statistical results.
    """

    fig, axes = plt.subplots(1, 3, figsize=(20, 15))

    # 统一字号设置
    label_fontsize = 20
    title_fontsize = 20
    legend_fontsize = 18

    # 1. Bootstrap distribution
    ax1 = axes[0]
    if 'bootstrap' in results:
        bootstrap_diffs = results['bootstrap']['bootstrap_diffs']
        ax1.hist(bootstrap_diffs, bins=30, alpha=0.7, color='steelblue', edgecolor='black')

        # 调整垂直线和图例
        ax1.axvline(results['bootstrap']['mean_difference'],
                    color='red', linestyle='--', linewidth=2, label='Mean')
        ax1.axvline(results['bootstrap']['ci_95'][0],
                    color='green', linestyle=':', linewidth=2, label='95% CI')
        ax1.axvline(results['bootstrap']['ci_95'][1],
                    color='green', linestyle=':', linewidth=2)

        ax1.set_xlabel('Mean difference', fontsize=label_fontsize)
        ax1.set_ylabel('Frequency', fontsize=label_fontsize)
        ax1.set_title('(a)Bootstrap Distribution\nof Mean Difference',
                      fontsize=title_fontsize, fontweight='bold')

        # 使用优化的图例函数
        legend1 = optimize_legend(ax1, fontsize=legend_fontsize, loc='upper right')

        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=18)

    # 2. Bayesian posterior distribution
    ax2 = axes[1]
    if 'bayesian' in results:
        post_mean = results['bayesian']['posterior_mean']
        post_sd = results['bayesian']['posterior_sd']

        x = np.linspace(post_mean - 3 * post_sd, post_mean + 3 * post_sd, 1000)
        y = stats.norm.pdf(x, loc=post_mean, scale=post_sd)

        ax2.plot(x, y, 'b-', linewidth=2, label='Posterior')
        ax2.fill_between(x, 0, y,
                         where=(x >= results['bayesian']['credible_95'][0]) &
                               (x <= results['bayesian']['credible_95'][1]),
                         color='blue', alpha=0.3, label='95% Credible Interval')
        ax2.axvline(0, color='red', linestyle='--',
                    linewidth=1, label='Null (difference=0)')

        ax2.set_xlabel('Mean Difference', fontsize=label_fontsize)
        ax2.set_ylabel('Probability Density', fontsize=label_fontsize)
        ax2.set_title('(b)Bayesian Posterior Distribution',
                      fontsize=title_fontsize, fontweight='bold')

        # 优化图例位置
        legend2 = ax2.legend(fontsize=legend_fontsize,
                             loc='upper right',
                             bbox_to_anchor=(0.80, 0.25),
                             frameon=True,
                             fancybox=True,
                             framealpha=0.9,
                             edgecolor='black')

        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=18)

    # 3. Effect size visualization
    ax3 = axes[2]
    cohen_d = results['power_analysis']['cohen_d']

    # 创建哑变量分布来可视化效应大小
    x = np.linspace(-3, 3, 1000)
    y_straight = stats.norm.pdf(x, loc=cohen_d / 2, scale=1)
    y_branched = stats.norm.pdf(x, loc=-cohen_d / 2, scale=1)

    ax3.plot(x, y_straight, 'b-', linewidth=2, label='Straight')
    ax3.plot(x, y_branched, 'r-', linewidth=2, label='Branched')
    ax3.fill_between(x, 0, y_straight, color='blue', alpha=0.3)
    ax3.fill_between(x, 0, y_branched, color='red', alpha=0.3)

    ax3.set_xlabel('Standardized Value', fontsize=label_fontsize)
    ax3.set_ylabel('Probability Density', fontsize=label_fontsize)
    ax3.set_title(f'(c)Effect Size Visualization\n(Cohen\'s d = {cohen_d:.2f})',
                  fontsize=title_fontsize, fontweight='bold')

    # 优化图例位置
    legend3 = ax3.legend(fontsize=legend_fontsize,
                         loc='upper right',
                         bbox_to_anchor=(1, 1),
                         frameon=True,
                         fancybox=True,
                         framealpha=0.9,
                         edgecolor='black')

    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', labelsize=11)

    # 调整整个图形的布局，为图例留出更多空间
    plt.tight_layout(rect=[0, 0, 0.95, 1])

    # 如果需要进一步调整，可以手动设置子图间距
    plt.subplots_adjust(wspace=0.3, right=0.92)

    plt.savefig("advanced_statistical_analysis.png", dpi=500, bbox_inches='tight')
    plt.show()


# ==================== INTERACTIVE VISUALIZATION ====================
def create_interactive_dashboard(straight_df, branched_df):
    """
    Create interactive dashboard using Plotly.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping interactive dashboard.")
        return None

    print("\nCreating interactive dashboard...")

    # Combine data
    straight_df['Chain Type'] = 'Straight'
    branched_df['Chain Type'] = 'Branched'
    combined_df = pd.concat([straight_df, branched_df], ignore_index=True)

    # 1. Interactive scatter plot
    fig1 = px.scatter(
        combined_df,
        x='total_CF',
        y='predicted_toxicity',
        color='Chain Type',
        hover_data=['SMILES', 'first_class', 'second_class', 'functional_group', 'MW', 'LogP'],
        title='Interactive Scatter: Toxicity vs CF Chain Length',
        labels={
            'total_CF': 'Total Fluorinated Carbons',
            'predicted_toxicity': 'Predicted Toxicity (pLD₅₀)'
        },
        color_discrete_map={'Straight': 'skyblue', 'Branched': 'lightcoral'}
    )

    # Add trend lines
    for chain_type in ['Straight', 'Branched']:
        subset = combined_df[combined_df['Chain Type'] == chain_type]
        if len(subset) > 1:
            x = subset['total_CF'].values.reshape(-1, 1)
            y = subset['predicted_toxicity'].values
            lr = LinearRegression().fit(x, y)
            x_range = np.linspace(x.min(), x.max(), 100)
            y_pred = lr.predict(x_range.reshape(-1, 1))

            fig1.add_trace(go.Scatter(
                x=x_range,
                y=y_pred,
                mode='lines',
                name=f'{chain_type} trend',
                line=dict(width=2, dash='dash'),
                showlegend=True
            ))

    # 2. Parallel coordinates plot
    available_cols = ['MW', 'LogP', 'TPSA', 'total_CF', 'predicted_toxicity']
    available_cols = [col for col in available_cols if col in combined_df.columns]

    if len(available_cols) >= 3:
        fig2 = px.parallel_coordinates(
            combined_df[available_cols + ['Chain Type']],
            color='Chain Type',
            dimensions=available_cols,
            color_continuous_scale=px.colors.diverging.Tealrose,
            title='Parallel Coordinates: Molecular Properties',
            labels={col: col.replace('_', ' ') for col in available_cols}
        )
    else:
        fig2 = None

    # 3. 3D scatter plot
    if all(col in combined_df.columns for col in ['MW', 'LogP', 'predicted_toxicity']):
        fig3 = px.scatter_3d(
            combined_df,
            x='MW',
            y='LogP',
            z='predicted_toxicity',
            color='Chain Type',
            hover_data=['SMILES', 'first_class', 'second_class', 'total_CF'],
            title='3D Visualization: MW vs LogP vs Toxicity',
            color_discrete_map={'Straight': 'blue', 'Branched': 'red'},
            opacity=0.7
        )
    else:
        fig3 = None

    # Save interactive plots
    fig1.write_html("interactive_scatter.html")
    print("  ✓ Saved interactive scatter plot as 'interactive_scatter.html'")

    if fig2:
        fig2.write_html("interactive_parallel_coordinates.html")
        print("  ✓ Saved parallel coordinates plot as 'interactive_parallel_coordinates.html'")

    if fig3:
        fig3.write_html("interactive_3d_scatter.html")
        print("  ✓ Saved 3D scatter plot as 'interactive_3d_scatter.html'")

    print("\nOpen the HTML files in a web browser for interactive visualization.")

    return {
        'scatter_plot': fig1,
        'parallel_coordinates': fig2,
        '3d_scatter': fig3
    }


# ==================== MOLECULAR STRUCTURE VISUALIZATION ====================
def visualize_molecular_structures_comparison(straight_df, branched_df, n_examples=5):
    """
    Visualize molecular structures for comparison.
    """
    print(f"\nVisualizing molecular structures ({n_examples} examples each)...")

    # Select representative molecules
    def select_representative_samples(df, n):
        if len(df) <= n:
            return df
        # Try to select diverse samples
        if 'total_CF' in df.columns and 'predicted_toxicity' in df.columns:
            # Stratified sampling based on CF length
            df = df.copy()
            df['cf_bin'] = pd.qcut(df['total_CF'], q=min(3, len(df)), labels=False, duplicates='drop')
            samples = []
            for bin_val in df['cf_bin'].unique():
                bin_samples = df[df['cf_bin'] == bin_val]
                if len(bin_samples) > 0:
                    samples.append(bin_samples.sample(min(1, len(bin_samples))))
            selected = pd.concat(samples, ignore_index=True)
            if len(selected) < n:
                # Add random samples to reach n
                remaining = df[~df.index.isin(selected.index)]
                if len(remaining) > 0:
                    additional = remaining.sample(min(n - len(selected), len(remaining)))
                    selected = pd.concat([selected, additional], ignore_index=True)
            return selected.head(n)
        else:
            return df.sample(min(n, len(df)))

    straight_samples = select_representative_samples(straight_df, n_examples)
    branched_samples = select_representative_samples(branched_df, n_examples)

    # Prepare molecules and legends
    mols = []
    legends = []

    for _, row in straight_samples.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol:
            mols.append(mol)
            toxicity = row.get('predicted_toxicity', 'N/A')
            cf_count = row.get('total_CF', 'N/A')
            legends.append(f"Straight\nTox: {toxicity:.2f}\nCF: {cf_count}")

    for _, row in branched_samples.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol:
            mols.append(mol)
            toxicity = row.get('predicted_toxicity', 'N/A')
            cf_count = row.get('total_CF', 'N/A')
            branch_pos = row.get('branch_position', 'N/A')
            legends.append(f"Branched ({branch_pos})\nTox: {toxicity:.2f}\nCF: {cf_count}")

    if not mols:
        print("No valid molecules found for visualization.")
        return None

    # Create grid image
    try:
        img = MolsToGridImage(
            mols,
            molsPerRow=min(5, len(mols)),
            subImgSize=(300, 300),
            legends=legends,
            returnPNG=False
        )

        # Save image
        img.save("molecular_structure_comparison.png")
        print("  ✓ Saved molecular structure comparison as 'molecular_structure_comparison.png'")

        # Display in notebook if available
        from IPython.display import Image, display
        display(Image(filename="molecular_structure_comparison.png"))

        return img
    except Exception as e:
        print(f"Error creating molecular visualization: {e}")
        return None


# ==================== GNN UTILITIES ====================
def load_pretrained_model(model_path):
    """
    Load pre-trained GNN model.
    """
    print(f"Loading pre-trained model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    model_params = {
        'hidden_dim': MODEL_CONFIG['params']['hidden_dim'],
        'num_layers': MODEL_CONFIG['params']['num_layers'],
        'dropout': MODEL_CONFIG['params']['dropout'],
        'prediction_tasks': 1,
        'input_dim': MODEL_CONFIG['params']['input_dim'],
        'model_type': MODEL_CONFIG['type_specific_params']['GNN']['model_type']
    }

    model = MoleculeGNN(**model_params)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully to {device}")
    return model


def get_or_predict_toxicity(df, model, device, smiles_col='SMILES'):
    """
    Get or predict toxicity values using GNN model.
    """
    print("Getting or predicting toxicity values...")

    # Try to load cached predictions
    predictions_df = pd.DataFrame()
    if os.path.exists(PREDICTIONS_CACHE_PATH):
        try:
            predictions_df = pd.read_csv(PREDICTIONS_CACHE_PATH)
            print(f"  Loaded {len(predictions_df)} cached predictions")
        except Exception as e:
            print(f"  Cache load error: {e}")
            predictions_df = pd.DataFrame()

    # Identify uncached molecules
    cached_smiles = set(predictions_df['SMILES']) if not predictions_df.empty else set()
    all_smiles = set(df[smiles_col])
    uncached_smiles = all_smiles - cached_smiles

    # Predict uncached molecules
    if uncached_smiles:
        print(f"  Predicting toxicity for {len(uncached_smiles)} uncached molecules...")

        graph_builder = MoleculeGraphBuilder()
        uncached_subset = df[df[smiles_col].isin(uncached_smiles)].copy()

        pyg_data_list = []
        uncached_smiles_list = []

        for _, row in uncached_subset.iterrows():
            smi = row[smiles_col]
            try:
                pyg_data = graph_builder.smiles_to_pyg_data(smi)
                if pyg_data is not None:
                    pyg_data_list.append(pyg_data)
                    uncached_smiles_list.append(smi)
            except Exception as e:
                print(f"  Error processing {smi}: {e}")

        if pyg_data_list:
            # Batch prediction
            data_loader = DataLoader(pyg_data_list, batch_size=min(64, len(pyg_data_list)))

            all_predictions = []
            with torch.no_grad():
                for batch in data_loader:
                    batch = batch.to(device)
                    pred, _ = model(batch.x, batch.edge_index, batch.batch)
                    pred_denorm = pred * TARGET_STD + TARGET_MEAN
                    all_predictions.extend(pred_denorm.cpu().numpy().flatten())

            # Create new predictions dataframe
            new_pred_df = pd.DataFrame({
                'SMILES': uncached_smiles_list,
                'predicted_toxicity': all_predictions
            })

            # Merge with existing cache
            predictions_df = pd.concat([predictions_df, new_pred_df], ignore_index=True)

            # Save updated cache
            predictions_df.to_csv(PREDICTIONS_CACHE_PATH, index=False)
            print(f"  Updated cache saved with {len(predictions_df)} predictions")

    # Merge predictions with original dataframe
    smiles_to_tox = dict(zip(predictions_df['SMILES'], predictions_df['predicted_toxicity']))
    df['predicted_toxicity'] = df[smiles_col].map(smiles_to_tox)

    # Check for missing predictions
    missing_count = df['predicted_toxicity'].isna().sum()
    if missing_count > 0:
        print(f"  Warning: {missing_count} molecules missing toxicity predictions")

    return df


# ==================== MAIN ANALYSIS PIPELINE ====================
def analyze_specific_combinations_enhanced(df_path, output_dir='enhanced_branched_analysis'):
    """
    Enhanced analysis pipeline for branched vs straight chain comparison.
    """
    print("=" * 70)
    print("ENHANCED PFAS BRANCHED VS STRAIGHT CHAIN ANALYSIS")
    print("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and prepare data
    print("\n1. Loading and preparing data...")
    df = pd.read_csv(df_path)
    print(f"   Loaded {len(df)} molecules")

    # Calculate total CF if not present
    if 'total_CF' not in df.columns:
        df['total_CF'] = df['CF2'].fillna(0) + df['CF3'].fillna(0) + df['CFX'].fillna(0)
        print("   Calculated total_CF from CF components")

    # 2. Filter target combinations
    print("\n2. Filtering target combinations...")
    mask = df[['first_class', 'second_class']].apply(
        lambda row: (row['first_class'], row['second_class']) in TARGET_COMBINATIONS,
        axis=1
    )
    target_df = df[mask].copy()

    if len(target_df) == 0:
        raise ValueError("No molecules match the specified combinations")

    print(f"   Found {len(target_df)} molecules in target combinations")

    # 3. Load GNN model and predict toxicity
    print("\n3. Loading GNN model and predicting toxicity...")
    model = load_pretrained_model(PRETRAINED_MODEL_PATH)
    target_df = get_or_predict_toxicity(target_df, model, device)

    # 4. Infer branch characteristics
    print("\n4. Inferring branch characteristics...")
    target_df['branch_position'] = target_df['SMILES'].apply(improved_branch_position_inference)

    branch_info = target_df['SMILES'].apply(infer_branch_length_and_complexity)
    target_df['branch_length'] = branch_info.apply(lambda x: x['length'])
    target_df['branch_complexity'] = branch_info.apply(lambda x: x['complexity'])
    target_df['branch_type'] = branch_info.apply(lambda x: x['type'])

    # 5. Split by chain type
    straight_df = target_df[target_df['chain_type'] == 'straight'].copy()
    branched_df = target_df[target_df['chain_type'] == 'branched'].copy()

    print(f"\n   Straight chains: {len(straight_df)} molecules")
    print(f"   Branched chains: {len(branched_df)} molecules")

    # 6. Sample balance analysis
    balance_report = analyze_sample_balance(straight_df, branched_df, TARGET_COMBINATIONS)

    # 7. Confounding factor analysis
    confounder_report = analyze_confounding_factors(straight_df, branched_df)

    # 8. Basic statistical tests
    print("\n" + "=" * 60)
    print("BASIC STATISTICAL COMPARISON")
    print("=" * 60)

    toxicity_col = 'predicted_toxicity'
    straight_tox = straight_df[toxicity_col].dropna().values
    branched_tox = branched_df[toxicity_col].dropna().values

    if len(straight_tox) > 0 and len(branched_tox) > 0:
        # Normality tests
        _, p_normal_straight = stats.shapiro(straight_tox)
        _, p_normal_branched = stats.shapiro(branched_tox)

        # Variance homogeneity test
        _, p_var = stats.levene(straight_tox, branched_tox)

        # Choose appropriate test
        if p_normal_straight > 0.05 and p_normal_branched > 0.05:
            print("   Both groups follow normal distribution")
            if p_var > 0.05:
                print("   Variances are equal, using Student's t-test")
                t_stat, p_value = stats.ttest_ind(straight_tox, branched_tox, equal_var=True)
                test_name = "Student's t-test"
            else:
                print("   Variances are unequal, using Welch's t-test")
                t_stat, p_value = stats.ttest_ind(straight_tox, branched_tox, equal_var=False)
                test_name = "Welch's t-test"
        else:
            print("   Non-normal distribution, using Mann-Whitney U test")
            u_stat, p_value = stats.mannwhitneyu(straight_tox, branched_tox, alternative='two-sided')
            test_name = "Mann-Whitney U test"

        # Effect size
        n1, n2 = len(straight_tox), len(branched_tox)
        pooled_std = np.sqrt(((n1 - 1) * np.var(straight_tox) + (n2 - 1) * np.var(branched_tox)) / (n1 + n2 - 2))
        cohen_d = (np.mean(straight_tox) - np.mean(branched_tox)) / pooled_std

        print(f"\n   Results:")
        print(f"   Test: {test_name}")
        print(f"   P-value: {p_value:.6f}")
        print(f"   Effect size (Cohen's d): {cohen_d:.3f}")
        print(f"   Straight mean: {np.mean(straight_tox):.3f} ± {np.std(straight_tox):.3f}")
        print(f"   Branched mean: {np.mean(branched_tox):.3f} ± {np.std(branched_tox):.3f}")
        print(f"   Difference: {np.mean(straight_tox) - np.mean(branched_tox):.3f}")

        # Statistical significance interpretation
        if p_value < 0.001:
            significance = "*** (p < 0.001)"
        elif p_value < 0.01:
            significance = "** (p < 0.01)"
        elif p_value < 0.05:
            significance = "* (p < 0.05)"
        else:
            significance = "not significant"

        print(f"\n   Statistical significance: {significance}")

        # Effect size interpretation
        if abs(cohen_d) < 0.2:
            effect_magnitude = "negligible"
        elif abs(cohen_d) < 0.5:
            effect_magnitude = "small"
        elif abs(cohen_d) < 0.8:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"

        print(f"   Effect magnitude: {effect_magnitude}")
    else:
        print("   Insufficient data for statistical comparison")
        p_value = None
        cohen_d = None
        test_name = None

    # 9. Advanced statistical analysis
    if len(straight_tox) > 10 and len(branched_tox) > 10:
        advanced_stats = perform_advanced_statistical_analysis(straight_tox, branched_tox)
    else:
        advanced_stats = None
        print("\nSkipping advanced statistical analysis due to small sample size")

    # 10. Multivariate analysis
    print("\n" + "=" * 60)
    print("MULTIVARIATE REGRESSION ANALYSIS")
    print("=" * 60)

    combined_df = pd.concat([straight_df, branched_df], ignore_index=True)
    multivariate_results = perform_multivariate_analysis(combined_df)

    # 11. Create visualizations
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)

    # Basic comparison plots
    from matplotlib import pyplot as plt

    # Toxicity comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot
    data = [straight_tox, branched_tox]
    labels = ['Straight', 'Branched']
    colors = ['skyblue', 'lightcoral']

    bp = axes[0].boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[0].set_ylabel('Predicted Toxicity (pLD₅₀)', fontsize=12)
    axes[0].set_title('Toxicity Comparison: Box Plot', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].tick_params(axis='both', which='major', labelsize=10)

    # Add significance annotation
    if p_value is not None and p_value < 0.05:
        y_max = max(straight_tox.max(), branched_tox.max())
        y_min = min(straight_tox.min(), branched_tox.min())
        y_range = y_max - y_min

        if p_value < 0.001:
            sig_symbol = '***'
        elif p_value < 0.01:
            sig_symbol = '**'
        else:
            sig_symbol = '*'

        axes[0].plot([1, 1, 2, 2],
                     [y_max + 0.1 * y_range, y_max + 0.15 * y_range,
                      y_max + 0.15 * y_range, y_max + 0.1 * y_range],
                     lw=1.5, c='black')
        axes[0].text(1.5, y_max + 0.2 * y_range, sig_symbol,
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Violin plot
    import seaborn as sns
    data_for_violin = pd.DataFrame({
        'Toxicity': np.concatenate([straight_tox, branched_tox]),
        'Chain Type': ['Straight'] * len(straight_tox) + ['Branched'] * len(branched_tox)
    })

    sns.violinplot(x='Chain Type', y='Toxicity', data=data_for_violin,
                   ax=axes[1], palette=colors, inner='quartile', cut=0)
    axes[1].set_title('Toxicity Comparison: Violin Plot', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('')  # 与左侧共享y轴标签
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].tick_params(axis='both', which='major', labelsize=10)

    # 调整整体布局
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "toxicity_comparison.png"),
                dpi=500, bbox_inches='tight')
    plt.show()

    # 12. Interactive visualizations
    interactive_plots = create_interactive_dashboard(straight_df, branched_df)

    # 13. Molecular structure visualization
    mol_vis = visualize_molecular_structures_comparison(straight_df, branched_df, n_examples=4)

    # 14. Generate comprehensive report
    print("\n" + "=" * 60)
    print("GENERATING COMPREHENSIVE REPORT")
    print("=" * 60)

    report_path = generate_comprehensive_report(
        straight_df=straight_df,
        branched_df=branched_df,
        p_value=p_value,
        cohen_d=cohen_d,
        test_name=test_name,
        balance_report=balance_report,
        confounder_report=confounder_report,
        multivariate_results=multivariate_results,
        advanced_stats=advanced_stats,
        output_dir=output_dir
    )

    print(f"\n✅ Analysis complete!")
    print(f"📊 Results saved to: {os.path.abspath(output_dir)}")
    print(f"📄 Comprehensive report: {report_path}")

    if interactive_plots:
        print("\n🌐 Interactive visualizations available as HTML files")
        print("   Open them in a web browser for interactive exploration")

    return {
        'straight_df': straight_df,
        'branched_df': branched_df,
        'p_value': p_value,
        'cohen_d': cohen_d,
        'test_name': test_name,
        'balance_report': balance_report,
        'confounder_report': confounder_report,
        'multivariate_results': multivariate_results,
        'advanced_stats': advanced_stats,
        'interactive_plots': interactive_plots,
        'output_dir': output_dir,
        'report_path': report_path
    }


def generate_comprehensive_report(straight_df, branched_df, p_value, cohen_d, test_name,
                                  balance_report, confounder_report, multivariate_results,
                                  advanced_stats, output_dir):
    """
    Generate a comprehensive analysis report.
    """
    report_path = os.path.join(output_dir, "comprehensive_analysis_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE PFAS BRANCHED VS STRAIGHT CHAIN ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("ANALYSIS SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target Combinations: {len(TARGET_COMBINATIONS)}\n")
        f.write(f"Total Molecules Analyzed: {len(straight_df) + len(branched_df)}\n")
        f.write(f"  - Straight Chains: {len(straight_df)}\n")
        f.write(f"  - Branched Chains: {len(branched_df)}\n")
        f.write(f"Output Directory: {output_dir}\n\n")

        f.write("STATISTICAL RESULTS\n")
        f.write("-" * 40 + "\n")
        if p_value is not None and cohen_d is not None:
            f.write(f"Statistical Test: {test_name}\n")
            f.write(f"P-value: {p_value:.6f}\n")
            f.write(f"Effect Size (Cohen's d): {cohen_d:.3f}\n")

            # Interpretation
            if p_value < 0.05:
                f.write("Conclusion: Significant difference found\n")
            else:
                f.write("Conclusion: No significant difference found\n")

            # Effect size interpretation
            if abs(cohen_d) < 0.2:
                effect_str = "negligible effect"
            elif abs(cohen_d) < 0.5:
                effect_str = "small effect"
            elif abs(cohen_d) < 0.8:
                effect_str = "medium effect"
            else:
                effect_str = "large effect"

            f.write(f"Effect Magnitude: {effect_str}\n")
        else:
            f.write("Insufficient data for statistical analysis\n")

        f.write("\nSAMPLE BALANCE ASSESSMENT\n")
        f.write("-" * 40 + "\n")
        if balance_report is not None:
            for _, row in balance_report.iterrows():
                f.write(f"{row['Combination']}: Straight={row['Straight']}, Branched={row['Branched']}, "
                        f"Ratio={row['Ratio']}, Imbalance={row['Imbalance']}\n")

        f.write("\nCONFOUNDING FACTOR ANALYSIS\n")
        f.write("-" * 40 + "\n")
        if confounder_report is not None:
            significant_confounders = confounder_report[confounder_report['Significant']]
            if len(significant_confounders) > 0:
                f.write("Significant Confounding Factors Found:\n")
                for _, row in significant_confounders.iterrows():
                    # 使用不同的引号来避免f-string中的反斜杠问题
                    factor = row['Factor']
                    p_val = row['P-Value']
                    cohens_d_val = row['Cohens_d']
                    f.write(f"  - {factor}: p={p_val}, d={cohens_d_val}\n")
                f.write("Recommendation: Control for these factors in analysis\n")
            else:
                f.write("No significant confounding factors detected\n")

        f.write("\nMULTIVARIATE ANALYSIS SUMMARY\n")
        f.write("-" * 40 + "\n")
        if multivariate_results is not None:
            rf_importance = multivariate_results['random_forest']['importance']
            f.write("Top 5 Most Important Features (Random Forest):\n")
            for _, row in rf_importance.head(5).iterrows():
                f.write(f"  - {row['Feature']}: Importance={row['Importance']:.3f}\n")

            if 'lasso' in multivariate_results:
                selected = multivariate_results['lasso']['selected_features']
                f.write(f"\nFeatures Selected by LASSO: {', '.join(selected)}\n")

        f.write("\nADVANCED STATISTICAL ANALYSIS\n")
        f.write("-" * 40 + "\n")
        if advanced_stats is not None:
            if 'bootstrap' in advanced_stats:
                ci = advanced_stats['bootstrap']['ci_95']
                f.write(f"Bootstrap 95% CI for mean difference: [{ci[0]:.3f}, {ci[1]:.3f}]\n")

            if 'bayesian' in advanced_stats:
                prob = advanced_stats['bayesian']['prob_positive']
                f.write(f"Bayesian probability that difference > 0: {prob:.1%}\n")

            if 'power_analysis' in advanced_stats:
                power = advanced_stats['power_analysis']['achieved_power']
                f.write(f"Statistical power achieved: {power:.1%}\n")

        f.write("\nKEY FINDINGS AND RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")

        if p_value is not None:
            if p_value < 0.05:
                f.write("1. Significant toxicity difference found between straight and branched chains.\n")
                if cohen_d > 0:
                    f.write("2. Straight chains appear to have higher toxicity than branched chains.\n")
                else:
                    f.write("2. Branched chains appear to have higher toxicity than straight chains.\n")

                if cohen_d is not None and abs(cohen_d) >= 0.5:
                    f.write("3. The effect size is clinically/scientifically meaningful.\n")
            else:
                f.write("1. No significant toxicity difference found between straight and branched chains.\n")
                f.write("2. Further investigation with larger sample size may be warranted.\n")

        f.write("\n3. Consider the following in future studies:\n")
        f.write("   - Larger sample sizes for increased statistical power\n")
        f.write("   - Control for identified confounding factors\n")
        f.write("   - Experimental validation of computational predictions\n")
        f.write("   - Investigation of specific branch position effects\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"  ✓ Comprehensive report saved to: {report_path}")
    return report_path


# ==================== MAIN ENTRY POINT ====================
if __name__ == "__main__":
    print("ENHANCED PFAS BRANCHED VS STRAIGHT CHAIN ANALYSIS SYSTEM")
    print("Version 2.0 - Comprehensive Statistical Analysis")
    print("=" * 60)

    try:
        # Run enhanced analysis
        results = analyze_specific_combinations_enhanced(DATA_PATH)

        if results:
            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 60)

            # Print key findings
            if results['p_value'] is not None:
                if results['p_value'] < 0.05:
                    print(f"\n✅ KEY FINDING: Significant difference found (p = {results['p_value']:.4f})")
                    print(f"   Effect size: Cohen's d = {results['cohen_d']:.3f}")

                    if results['cohen_d'] > 0:
                        print(f"   Interpretation: Straight chains are MORE toxic than branched chains")
                    else:
                        print(f"   Interpretation: Branched chains are MORE toxic than straight chains")
                else:
                    print(f"\n🔍 KEY FINDING: No significant difference found (p = {results['p_value']:.4f})")
                    print(f"   Effect size: Cohen's d = {results['cohen_d']:.3f}")

            print(f"\n📁 All results saved to: {results['output_dir']}")
            print(f"📄 Comprehensive report: {results['report_path']}")

            if results['interactive_plots']:
                print(f"\n🌐 Interactive visualizations available as HTML files")
                print("   Open them in a web browser for interactive exploration")

    except Exception as e:
        print(f"\n❌ ERROR: Analysis failed")
        print(f"   Error message: {e}")
        import traceback

        traceback.print_exc()

        # Try simplified analysis
        print("\nAttempting simplified analysis...")
        try:
            # Load data
            df = pd.read_csv(DATA_PATH)
            print(f"Loaded {len(df)} molecules")

            # Filter target combinations
            # mask = df[['first_class', 'second_class']].apply(
            #     lambda row: (row['first_class'], row['second_class']) in TARGET_COMBINATIONS,
            #     axis=1
            # )
            # target_df = df[mask]
            target_df = df

            if len(target_df) > 0:
                print(f"Found {len(target_df)} molecules in target combinations")

                # Simple split and comparison
                straight_df = target_df[target_df['chain_type'] == 'straight']
                branched_df = target_df[target_df['chain_type'] == 'branched']

                print(f"Straight: {len(straight_df)}, Branched: {len(branched_df)}")

                # Simple visualization
                if 'predicted_toxicity' in target_df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.boxplot([straight_df['predicted_toxicity'].dropna(),
                                 branched_df['predicted_toxicity'].dropna()],
                                labels=['Straight', 'Branched'])
                    plt.ylabel('Predicted Toxicity')
                    plt.title('Simplified Toxicity Comparison')
                    plt.grid(True, alpha=0.3)
                    plt.savefig('simplified_comparison.png', dpi=500, bbox_inches='tight')
                    plt.show()

                    print("\nSimplified analysis completed")
                    print("Check 'simplified_comparison.png' for results")

        except Exception as e2:
            print(f"Simplified analysis also failed: {e2}")