import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from matplotlib.widgets import Cursor

# ----------------------------
# Utility function to clean filename
# ----------------------------
def clean_filename(name):
    """Remove invalid characters from filename"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name

# ----------------------------
# Step 1: Load and merge data
# ----------------------------
df_smiles = pd.read_csv(r'F:\GNN-pro\data_process_class\step3_OECD_Class_Enhanced_v2_with_CF.csv')
df_toxicity = pd.read_csv(r'F:\GNN-pro\data_process_class\predict_oecd_cf2_GIN510_v3.csv')
# df = df_smiles.merge(df_toxicity, on='SMILES', how='inner')

# Manually define high R² second_class list (based on R² > 0.9 groups in PDF)
# high_r2_list = ['HFCs', 'PFAIs', 'PFAenes', 'PolyFACs', 'SFAenes', 'PFCA-ester derivatives', 'PolyFCAs']
high_r2_list = [
    'SFAenes',                              # r = 0.996
    'PFAenes',                              # r = 0.985
    'HFEs',                                 # r = 0.954
    'PFECAs',                               # r = 0.948
    'PFAIs',                                # r = 0.945
    'HFCs',                                 # r = 0.935
    'PolyFCAs',                             # r = 0.910
    'PFCA-ester derivatives',               # r = 0.899
    'Polyfluoroalkanes',                    # r = 0.878
    'PFPEs',                                # r = 0.852
    'PolyFEACs',                            # r = 0.806
    'n:2 fluorotelomer-based substances',   # r = 0.776
    'PASF-based substances',                # r = 0.764
    'PFAK derivatives',                     # r = 0.718
    'PolyFACs'                              # r = 0.701
]
# Filter high R² groups
df_high = df_toxicity[df_toxicity['second_class'].isin(high_r2_list)].copy()


print(f"Number of molecules in high R² groups: {len(df_high)}")

# ----------------------------
# Step 2: Calculate F/C Ratio using existing columns
# ----------------------------
# Calculate total fluorine atoms: CF2*2 + CF3*3 + CFX*1
df_high['Total_F'] = df_high['CF2'] * 2 + df_high['CF3'] * 3 + df_high.get('CFX', 0) * 1
df_high['Total_C'] = df_high['cf_length']  # Assuming cf_length is the number of carbon atoms in the fluorinated chain

# Avoid division by zero
df_high = df_high[df_high['Total_C'] > 0].copy()

# ----------------------------
# Step 3: Calculate TPSA, MW, and LogP from SMILES
# ----------------------------
def calc_descriptors(smiles):
    """Calculate molecular descriptors: TPSA, MW, and LogP"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan, np.nan, np.nan
    tpsa = Descriptors.TPSA(mol)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    return tpsa, mw, logp

print("Calculating TPSA, MW, and LogP...")
df_high[['TPSA', 'MW', 'LogP']] = df_high['SMILES'].apply(
    lambda s: pd.Series(calc_descriptors(s))
)

# Clean invalid values
required_cols = ['TPSA', 'MW', 'LogP', 'predicted_toxicity', 'cf_length']
df_high = df_high.dropna(subset=required_cols).reset_index(drop=True)
print(f"Remaining after cleaning: {len(df_high)}")

# Use 'cf_length' as CF chain length
df_high['CF_Length'] = df_high['cf_length']

# ----------------------------
# Step 4: Group-wise correlation heatmaps
# ----------------------------
plots_dir = 'logp_plots'
os.makedirs(plots_dir, exist_ok=True)

for second_cls in high_r2_list:
    group_data = df_high[df_high['second_class'] == second_cls]
    if len(group_data) < 5:
        continue

    # Include LogP in correlation analysis
    corr_vars = ['predicted_toxicity', 'CF_Length', 'TPSA', 'MW', 'LogP']
    corr_df = group_data[corr_vars]
    corr_matrix = corr_df.corr()

    plt.figure(figsize=(8, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.3f', square=True)
    plt.title(f'Correlation Heatmap\nGroup: {second_cls} (n={len(group_data)})')
    plt.tight_layout()

    # Clean filename to avoid invalid characters
    safe_filename = clean_filename(second_cls)
    plt.savefig(os.path.join(plots_dir, f'{safe_filename}_correlation_heatmap.png'),
                dpi=500, bbox_inches='tight')
    plt.close()

# ----------------------------
# Step 5: Global correlation + regression analysis
# ----------------------------
all_high = df_high.copy()

# Calculate global Pearson correlation coefficients
corrs = {}
for var in ['CF_Length', 'TPSA', 'MW', 'LogP']:
    r, p = stats.pearsonr(all_high['predicted_toxicity'], all_high[var])
    corrs[var] = {'Pearson_r': r, 'p-value': p}

corr_summary = pd.DataFrame(corrs).T
print("\n=== Global Correlation (All High r Groups Combined) ===")
print(corr_summary)

# Univariate regression
reg_results = []
for var in ['CF_Length', 'TPSA', 'MW', 'LogP']:
    X = all_high[[var]].values
    y = all_high['predicted_toxicity'].values
    model = LinearRegression().fit(X, y)
    score = model.score(X, y)
    coef = model.coef_[0]
    reg_results.append({
        'Predictor': var,
        'R²': score,
        'Coefficient': coef,
        'Pearson_r': corr_summary.loc[var, 'Pearson_r']
    })

reg_df = pd.DataFrame(reg_results)
print("\n=== Univariate Regression Results ===")
print(reg_df)

# Multivariate regression analysis: CF_Length vs CF_Length + LogP
print("\n=== Multivariate Regression Analysis (CF_Length vs CF_Length + LogP) ===")

# Model 1: CF_Length only
model_cf = LinearRegression().fit(all_high[['CF_Length']].values, all_high['predicted_toxicity'].values)
r2_cf_only = model_cf.score(all_high[['CF_Length']].values, all_high['predicted_toxicity'].values)

# Model 2: CF_Length + LogP
model_cf_logp = LinearRegression().fit(all_high[['CF_Length', 'LogP']].values, all_high['predicted_toxicity'].values)
r2_cf_logp = model_cf_logp.score(all_high[['CF_Length', 'LogP']].values, all_high['predicted_toxicity'].values)

# Model 3: LogP only
model_logp_only = LinearRegression().fit(all_high[['LogP']].values, all_high['predicted_toxicity'].values)
r2_logp_only = model_logp_only.score(all_high[['LogP']].values, all_high['predicted_toxicity'].values)

print(f"Model 1 (CF_Length only) R²: {r2_cf_only:.3f}")
print(f"Model 2 (CF_Length + LogP) R²: {r2_cf_logp:.3f}")
print(f"Model 3 (LogP only) R²: {r2_logp_only:.3f}")

# Save results
corr_summary.to_csv('new_mechanism_global_correlation.csv', encoding='utf-8-sig')
reg_df.to_csv('new_mechanism_univariate_regression.csv', encoding='utf-8-sig')

# ----------------------------
# Step 6: Group-wise R² comparison
# ----------------------------
group_r2 = []
for second_cls in high_r2_list:
    g = df_high[df_high['second_class'] == second_cls]
    if len(g) < 5:
        continue

    # Calculate R² for each variable
    r2_cf = LinearRegression().fit(g[['CF_Length']].values, g['predicted_toxicity'].values).score(
        g[['CF_Length']].values, g['predicted_toxicity'].values)
    r2_tpsa = LinearRegression().fit(g[['TPSA']].values, g['predicted_toxicity'].values).score(g[['TPSA']].values, g[
        'predicted_toxicity'].values)
    r2_logp = LinearRegression().fit(g[['LogP']].values, g['predicted_toxicity'].values).score(g[['LogP']].values, g[
        'predicted_toxicity'].values)

    # Calculate R² for CF_Length + LogP combination
    if len(g) >= 3:
        r2_cf_logp = LinearRegression().fit(g[['CF_Length', 'LogP']].values, g['predicted_toxicity'].values).score(
            g[['CF_Length', 'LogP']].values, g['predicted_toxicity'].values)
    else:
        r2_cf_logp = np.nan

    group_r2.append({
        'Group': second_cls,
        'R²(CF_Length)': r2_cf,
        'R²(LogP)': r2_logp,
        'R²(CF+LogP)': r2_cf_logp,
        'R²(TPSA)': r2_tpsa,
        'n': len(g)
    })

group_r2_df = pd.DataFrame(group_r2)
print("\n=== Group-wise R² Comparison ===")
print(group_r2_df)
group_r2_df.to_csv('new_mechanism_group_r2_comparison.csv', encoding='utf-8-sig')

# ----------------------------
# Step 7: Triple relationship analysis: CF_Length - LogP - Toxicity
# ----------------------------
print("\n=== Triple Relationship Analysis: CF_Length - LogP - Toxicity ===")

plt.figure(figsize=(8, 6))

# 1. CF_Length vs LogP
plt.subplot(2, 2, 1)
scatter1 = plt.scatter(all_high['CF_Length'], all_high['LogP'],
                       c=all_high['predicted_toxicity'], cmap='viridis', alpha=0.7, s=50)
plt.xlabel('Fluorinated Carbon Chain Length')
plt.ylabel('LogP')
plt.title('(a)Fluorinated Carbon Chain Length \nvs LogP (Color = Toxicity)')
plt.colorbar(scatter1, label='LD₅₀(-log(mol/kg)')

# Add trend line
if len(all_high) >= 2:
    z1 = np.polyfit(all_high['CF_Length'], all_high['LogP'], 1)
    p1 = np.poly1d(z1)
    x_fit1 = np.linspace(all_high['CF_Length'].min(), all_high['CF_Length'].max(), 100)
    plt.plot(x_fit1, p1(x_fit1), 'r-', linewidth=2)
    r1, p1_val = stats.pearsonr(all_high['CF_Length'], all_high['LogP'])
    plt.text(0.05, 0.95, f' Pearson $r = {r1:.3f}$', transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 2. LogP vs Toxicity
plt.subplot(2, 2, 2)
scatter2 = plt.scatter(all_high['LogP'], all_high['predicted_toxicity'],
                       c=all_high['CF_Length'], cmap='plasma', alpha=0.7, s=50)
plt.xlabel('LogP')
plt.ylabel('LD₅₀(-log(mol/kg)')
plt.title('(b)LogP vs LD₅₀(-log(mol/kg)')
plt.colorbar(scatter2, label='Fluorinated Carbon Chain Length')

if len(all_high) >= 2:
    z2 = np.polyfit(all_high['LogP'], all_high['predicted_toxicity'], 1)
    p2 = np.poly1d(z2)
    x_fit2 = np.linspace(all_high['LogP'].min(), all_high['LogP'].max(), 100)
    plt.plot(x_fit2, p2(x_fit2), 'r-', linewidth=2)
    r2, p2_val = stats.pearsonr(all_high['LogP'], all_high['predicted_toxicity'])
    plt.text(0.05, 0.95, f'Pearson $r = {r2:.3f}$', transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 3. CF_Length vs Toxicity
plt.subplot(2, 2, 3)
scatter3 = plt.scatter(all_high['CF_Length'], all_high['predicted_toxicity'],
                       c=all_high['LogP'], cmap='coolwarm', alpha=0.7, s=50)
plt.xlabel('Fluorinated Carbon Chain Length')
plt.ylabel('LD₅₀(-log(mol/kg)')
plt.title('(c)Fluorinated Carbon Chain Length \nvs LD₅₀(-log(mol/kg)')
plt.colorbar(scatter3, label='LogP')

if len(all_high) >= 2:
    z3 = np.polyfit(all_high['CF_Length'], all_high['predicted_toxicity'], 1)
    p3 = np.poly1d(z3)
    x_fit3 = np.linspace(all_high['CF_Length'].min(), all_high['CF_Length'].max(), 100)
    plt.plot(x_fit3, p3(x_fit3), 'r-', linewidth=2)
    r3, p3_val = stats.pearsonr(all_high['CF_Length'], all_high['predicted_toxicity'])
    plt.text(0.05, 0.95, f'Pearson $r = {r3:.3f}$', transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 4. LogP distribution histogram
plt.subplot(2, 2, 4)
plt.hist(all_high['LogP'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('LogP')
plt.ylabel('Frequency')
plt.title('(d)LogP Distribution')
plt.axvline(x=all_high['LogP'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {all_high["LogP"].mean():.2f}')

plt.legend()



# plt.suptitle('CF_Length - LogP - Toxicity Triple Relationship Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'chain_logp_toxicity_triple_analysis.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# Print statistical analysis results
print("\n=== CF_Length - LogP - Toxicity Statistical Analysis ===")
print(f"1. CF_Length vs LogP correlation: r = {r1:.3f} (p = {p1_val:.2e})")
print(f"2. LogP vs Toxicity correlation: r = {r2:.3f} (p = {p2_val:.2e})")
print(f"3. CF_Length vs Toxicity correlation: r = {r3:.3f} (p = {p3_val:.2e})")

# Relative contribution analysis of CF_Length and LogP
if r2_cf_logp > max(r2_cf_only, r2_logp_only):
    improvement = r2_cf_logp - max(r2_cf_only, r2_logp_only)
    print(
        f"\nThe combined model (CF_Length + LogP) improves R² by {improvement:.3f} compared to the best univariate model")
    print("This indicates that CF_Length and LogP provide independent information for toxicity prediction")

# ----------------------------
# Step 8: Relationship between LogP and Toxicity in different PFAS classes (Interactive Version)
# ----------------------------
print("\n=== Interactive Relationship between LogP and Toxicity in Different PFAS Classes ===")

# Create a figure with subplots
fig, axes = plt.subplots(2, 4, figsize=(12, 8))
axes = axes.flatten()

# Store scatter plots for interaction
scatter_plots = []

for idx, second_cls in enumerate(high_r2_list[:8]):  # Show up to 8 classes
    if idx >= len(axes):
        break

    group_data = df_high[df_high['second_class'] == second_cls]
    if len(group_data) < 3:
        axes[idx].text(0.5, 0.5, f'Insufficient data\n(n={len(group_data)})',
                       ha='center', va='center', transform=axes[idx].transAxes)
        axes[idx].set_title(f'{second_cls}')
        continue

    ax = axes[idx]
    # Create scatter plot with num_id as an annotation
    scatter = ax.scatter(group_data['LogP'], group_data['predicted_toxicity'],
                         alpha=0.7, s=40, picker=True)  # Enable picking
    scatter_plots.append((scatter, group_data))  # Store for callback

    # Add trend line
    if len(group_data) >= 2:
        z = np.polyfit(group_data['LogP'], group_data['predicted_toxicity'], 1)
        p = np.poly1d(z)
        x_fit = np.linspace(group_data['LogP'].min(), group_data['LogP'].max(), 100)
        ax.plot(x_fit, p(x_fit), 'r-', linewidth=2)

        # Calculate R²
        r, p_val = stats.pearsonr(group_data['LogP'], group_data['predicted_toxicity'])
        ax.text(0.05, 0.95, f'R² = {r ** 2:.3f}', transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('LogP')
    ax.set_ylabel('LD₅₀(-log(mol/kg)')
    ax.set_title(f'{second_cls} (n={len(group_data)})')
    ax.grid(True, alpha=0.3)

# Hide extra subplots
for idx in range(len(high_r2_list[:8]), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Relationship between LogP and Toxicity in Different PFAS Classes', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Define hover event handler
def on_hover(event):
    if event.inaxes is None:
        return
    for scatter, data in scatter_plots:
        if event.inaxes == scatter.axes:
            # Get the index of the point under cursor
            cont, ind = scatter.contains(event)
            if cont:
                # Get the first index
                i = ind["ind"][0]
                # Get molecule info
                mol_info = data.iloc[i]
                # Create annotation text
                text = f"num_id: {mol_info['num_id']}\nSMILES: {mol_info['SMILES']}\nToxicity: {mol_info['predicted_toxicity']:.3f}"
                # Remove previous annotation
                if hasattr(on_hover, 'annot'):
                    on_hover.annot.remove()
                # Create new annotation
                on_hover.annot = ax.annotate(text, xy=(event.xdata, event.ydata),
                                             xytext=(20, 20), textcoords="offset points",
                                             bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
                                             arrowprops=dict(arrowstyle="->"))
                fig.canvas.draw_idle()
            else:
                # Remove annotation if no point is under cursor
                if hasattr(on_hover, 'annot'):
                    on_hover.annot.remove()
                    fig.canvas.draw_idle()

# Connect the event
fig.canvas.mpl_connect('motion_notify_event', on_hover)

# Save the interactive plot (note: interactivity won't work in saved image, but you can save as HTML or use in Jupyter)
plt.savefig(os.path.join(plots_dir, 'logp_toxicity_by_category_interactive.png'),
            dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

# ----------------------------
# Step 9: Global Correlation Heatmap for All Selected Molecules
# ----------------------------
print("\n=== Generating Global Correlation Heatmap for All Selected Molecules ===")

# Define variables and their readable labels
corr_vars = ['predicted_toxicity', 'CF_Length', 'TPSA', 'MW', 'LogP']
readable_labels = [
    'LD₅₀(-log(mol/kg))',
    'Fluorinated Carbon\nChain Length',
    'TPSA',
    'MW',
    'LogP'
]

# Select data and compute correlation matrix
corr_df = all_high[corr_vars]
corr_matrix = corr_df.corr()

# Create the heatmap
plt.figure(figsize=(20, 16))
ax = sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    center=0,
    fmt='.3f',
    square=True,
    linewidths=.5,
    xticklabels=readable_labels,
    yticklabels=readable_labels,
    cbar_kws={"shrink": .8}  # 确保 colorbar 被正确创建
)

# --- 修改开始：安全地获取 colorbar 并设置字体 ---
# 方法：通过 ax.collections 获取 QuadMesh 对象，再获取 colorbar
quad_mesh = ax.collections[0]  # heatmap 使用的是 QuadMesh
cbar = quad_mesh.colorbar

if cbar is not None:
    cbar.ax.tick_params(labelsize=18)  # 设置 colorbar 刻度字体大小
# --- 修改结束 ---

# 设置坐标轴标签字体大小
ax.set_xticklabels(readable_labels, fontsize=20, rotation=45, ha='right')
ax.set_yticklabels(readable_labels, fontsize=20, rotation=0)

# 设置标题
# plt.title(f'Global Correlation Heatmap\nStrongly Correlated Subclasses (n={len(all_high)})', fontsize=20, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'global_correlation_heatmap_all_selected.png'),
            dpi=500, bbox_inches='tight')
plt.show()
