import pandas as pd
from rdkit import Chem

def count_cf_groups(smiles_str):
    """
    Counts the number of -CF2- groups using SMARTS pattern matching.
    This is a more robust way to find specific structural motifs.
    """
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            print(f"Warning: Could not parse SMILES: {smiles_str}")
            return {'CF2': -1, 'CF3': -1, 'CFX': -1}

        CF2_count = 0
        CF3_count = 0
        CFX_count = 0

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6: # 找到碳原子
                neighbors = atom.GetNeighbors()
                F_count = 0
                C_count = 0
                H_count = 0
                O_count = 0 # Add count for O, S, etc. if needed for bridge logic later
                for neighbor in neighbors:
                    neighbor_atomic_num = neighbor.GetAtomicNum()
                    if neighbor_atomic_num == 9: # 氟
                        F_count += 1
                    elif neighbor_atomic_num == 6: # 碳
                        C_count += 1
                    elif neighbor_atomic_num == 1: # 氢
                        H_count += 1
                    elif neighbor_atomic_num == 8: # 氧
                        O_count += 1

                # Original logic (strict direct neighbor check)
                if F_count == 2 :
                    CF2_count += 1
                    # print(f"Debug: Atom {atom.GetIdx()} (SMILES: {smiles_str[:30]}...) counted as CF2: F={F_count}, C={C_count}")
                elif F_count == 3:
                    CF3_count += 1
                elif F_count == 1:
                    CFX_count += 1

        return {'CF2': CF2_count, 'CF3': CF3_count, 'CFX': CFX_count}

    except Exception as e:
        print(f"Error processing SMILES: {smiles_str}, Error: {e}")
        return {'CF2': -1, 'CF3': -1, 'CFX': -1}

# --- 1. 加载原始数据 ---
print("正在加载原始数据...")
df = pd.read_csv('F:\GNN-pro\gen_data\Atlas_all.csv', header=None)
# 根据知识库推断列名
df.columns = ['RDKIT_SMILES', 'SMILES', 'MHFP', 'first_class', 'second_class']
print(f"原始数据加载完成，共 {len(df)} 行。")

# --- 2. 定义结构特征提取函数 (已融合 CF 计数) ---
def analyze_structure(smiles_str):
    """
    Analyze a SMILES string to extract structural features AND CF group counts.
    Returns a dictionary containing chain_type, carbon_chain_length, ring_size, functional_group, CF2, CF3, CFX.
    """
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return {
                'chain_type': 'unparsable',
                'carbon_chain_length': None,
                'ring_size': None,
                'functional_group': 'Unknown',
                'CF2': -1,
                'CF3': -1,
                'CFX': -1
            }

        # Initialize features
        chain_type = 'unclassified'
        carbon_chain_length = None
        ring_size = None
        functional_group = 'Unknown'
        CF2, CF3, CFX = -1, -1, -1 # Initialize counts

        # --- 1. Determine Ring Size (if any) ---
        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() > 0:
            ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
            if ring_sizes:
                ring_size = min(ring_sizes)
            chain_type = 'cyclic'
        else:
            chain_type = 'acyclic'

        # --- 2. Analyze Carbon Chain for Acyclic Molecules ---
        if chain_type == 'acyclic':
            # Count the total number of carbon atoms as a proxy for the main carbon framework size.
            # This is a simplification but works well for many PFAS.
            carbon_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
            carbon_count = len(carbon_atoms)
            carbon_chain_length = carbon_count

            # Determine if it's straight or branched
            branch_points = 0
            for atom in carbon_atoms:
                num_c_neighbors = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 6)
                if num_c_neighbors > 2: # If a carbon is connected to more than 2 other carbons
                    branch_points += 1
            if branch_points > 0:
                chain_type = 'branched'
            else:
                chain_type = 'straight'

        # --- 3. Extract Functional Group ---
        # Use the second_class as a primary source for functional group
        # This is a heuristic based on common patterns in the data.
        FUNCTIONAL_GROUP_RULES = {
            "Carboxylic Acid": "C(=O)[O;H1]",      # 羧酸: 连接羟基的羰基碳
            "Ester": "C(=O)[O;H0;!$(O=C)]",        # 酯: 连接非羰基氧的羰基碳 (排除羧酸)
            "Aldehyde": "[CH1](=O)",               # 醛: 连接氢的羰基碳
            "Ketone": "[C;!H0](=O)",               # 酮: 不连接氢的羰基碳 (排除醛)
            "Alcohol": "[O;H1][C;!$(C=O)]",        # 醇: 连接非羰基碳的羟基氧 (排除羧酸)
            "Ether": "[O;H0]([C,c])[C,c]",         # 醚: 连接两个碳原子的氧
            "Amine": "[N;H2,H1,H0][C,c]",          # 胺: 连接碳原子的氮 (伯、仲、叔胺)
            "Amide": "C(=O)[N;H2,H1,H0]",          # 酰胺: 连接氮的羰基碳
            "Halogen": "[Cl,Br,I]",              # 卤素原子
            "Sulfonic Acid": "S(=O)(=O)[O;H1]",     # 磺酸: 连接羟基的磺酰基硫
            "Sulfonate Ester": "S(=O)(=O)[O;H0;!$(O=S)]", # 磺酸酯: 连接非磺酰基氧的磺酰基硫 (排除磺酸)
            "Phosphate": "P(=O)([O;H1,H0])([O;H1,H0])[O;H1,H0]", # 磷酸基 (较宽松，可能需要细化)
            "Phosphonate Ester": "P(=O)([O;H0][C,c])([O;H0][C,c])[O;H0][C,c]", # 磷酸酯 (较宽松，可能需要细化)
            "Thiol": "[S;H1][C,c]",                # 巯基: 连接碳的硫醇氢
            "Thioether": "[S;H0]([C,c])[C,c]",      # 硫醚: 连接两个碳的硫 (排除亚砜、砜)
            "Disulfide": "[S;H0][S;H0]",           # 二硫键
            "Nitro": "N(=O)=O",                    # 硝基
            "Nitrile": "C#N",                      # 腈基
            "Isocyanate": "N=C=O",                 # 异氰酸酯
            "Carbamate": "N-C(=O)-O",              # 氨基甲酸酯 (较宽松)
            "Sulfone": "S(=O)(=O)",                # 砜
            "Sulfoxide": "S(=O)",                  # 亚砜  # 酰胺 (C(=O)N 连接模式)
            "Aromatic": "a",  # 芳香环上的原子 (小写 a)

        }

        def identify_functional_group_rdkit(smiles_str):
            """
            使用 RDKit 和 SMARTS 规则识别分子的主要官能团。
            优先返回更具体的官能团。
            """
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                return "Unparsable_SMILES"

            # 按优先级顺序检查官能团
            # 例如，如果既有 COOH 又有 Amide，可能需要定义哪个优先级更高
            # 这里我们简单地返回第一个匹配到的，您可以根据需要调整顺序
            for group_name, smarts_pattern in FUNCTIONAL_GROUP_RULES.items():
                if group_name == "Aromatic":
                    # 对于芳香性，检查分子中是否包含芳香原子
                    # 使用 'a' SMARTS 模式
                    pattern_mol = Chem.MolFromSmarts(smarts_pattern)
                    if pattern_mol is not None and mol.HasSubstructMatch(pattern_mol):
                        return group_name
                else:
                    # 对于其他 SMARTS 模式，如 COOH, SO3H, I, Amide
                    pattern_mol = Chem.MolFromSmarts(smarts_pattern)
                    if pattern_mol is not None and mol.HasSubstructMatch(pattern_mol):
                        return group_name

            # 如果没有匹配到任何规则，则返回 "Unknown"
            return "Unknown"
        functional_group = identify_functional_group_rdkit(smiles_str)

        # --- 4. Count CF Groups (Integrated here) ---
        cf_counts = count_cf_groups(smiles_str)
        CF2 = cf_counts['CF2']
        CF3 = cf_counts['CF3']
        CFX = cf_counts['CFX']

        return {
            'chain_type': chain_type,
            'carbon_chain_length': carbon_chain_length,
            'ring_size': ring_size,
            'functional_group': functional_group,
            'CF2': CF2,
            'CF3': CF3,
            'CFX': CFX
        }

    except Exception as e:
        print(f"Error analyzing SMILES: {smiles_str}, Error: {e}")
        return {
            'chain_type': 'error',
            'carbon_chain_length': None,
            'ring_size': None,
            'functional_group': 'Unknown',
            'CF2': -1,
            'CF3': -1,
            'CFX': -1
        }

# --- 3. Apply the analysis function to each row ---
print("正在分析分子结构特征...")
struct_features = df['SMILES'].apply(analyze_structure)
features_df = pd.DataFrame(struct_features.tolist())

# --- 4. Combine original data with new features ---
print("正在合并数据...")
enhanced_df = pd.concat([df, features_df], axis=1)

# --- 5. Generate the NEW group_id (including CF group counts) ---
# Create a group ID that incorporates all key variables for control-variable analysis, including CF counts
# Handle NaN values by converting to string and replacing with 'NA'
enhanced_df['group_id'] = (
    enhanced_df['first_class'].astype(str) + '_' +
    enhanced_df['second_class'].astype(str) + '_' +
    enhanced_df['chain_type'].astype(str) + '_' +
    enhanced_df['carbon_chain_length'].fillna('NA').astype(str) + '_' +
    enhanced_df['functional_group'].astype(str) + '_' +
    enhanced_df['ring_size'].fillna('NA').astype(str) + '_' +
    'CF2_' + enhanced_df['CF2'].astype(str) + '_CF3_' + enhanced_df['CF3'].astype(str) + '_CFX_' + enhanced_df['CFX'].astype(str)
)

# --- 6. Save the enhanced data table ---
output_filename = 'F:\GNN-pro\gen_data\generated_253_molecules_enhanced.csv'
print(f"正在保存增强数据表到 {output_filename}...")
enhanced_df.to_csv(output_filename, index=False)
print(f"增强数据表已保存完成！")

# --- 7. Demonstrate the new group_id ---
subset = enhanced_df[
    (enhanced_df['first_class'] == 'Other PFASs') &
    (enhanced_df['second_class'] == 'Aromatic PFASs')
    # (enhanced_df['carbon_chain_length'] >= 8.0) &  # 大于等于 4.0
    # (enhanced_df['carbon_chain_length'] <= 12.0)    # 小于等于 8.0
]
print(subset)
subset_chain_type = subset[subset['chain_type'] == 'cyclic']#branched、 straight、 cyclic
print(subset_chain_type)
subset_cf = subset_chain_type[
    (subset_chain_type['CF2'] <= 6)&
    (subset_chain_type['CF2'] >= 0)
    # (subset_chain_type['CF3'] <= 6)&
    # (subset_chain_type['CFX'] <= 6)
]
print(subset_cf)



# print("\n筛选结果 (Other PFASs - Aromatic PFASs, cyclic, C8-C12):")
# print(subset_chain_type[['SMILES', 'first_class', 'second_class', 'chain_type', 'carbon_chain_length', 'CF2', 'CF3', 'CFX', 'group_id']])
#
# # --- 8. Display summary ---
# print("\n--- 数据增强摘要 ---")
# print(f"原始数据行数: {len(df)}")
# print(f"增强后数据行数: {len(enhanced_df)}")
# print(f"新增特征列: ['chain_type', 'carbon_chain_length', 'ring_size', 'functional_group', 'CF2', 'CF3', 'CFX', 'group_id']")
# print("\n新增列的统计信息 (CF groups):")
# print(enhanced_df[['CF2', 'CF3', 'CFX']].describe())
# print(f"\n无法解析的分子数量 (CF计数为-1): {(enhanced_df['CF2'] == -1).sum()}")
#
# print("\n新增列的示例值 (前10行):")
# print(enhanced_df[['SMILES', 'first_class', 'second_class', 'chain_type', 'carbon_chain_length', 'ring_size', 'functional_group', 'CF2', 'CF3', 'CFX', 'group_id']].head(10))
#
# # Show unique group IDs for a quick overview
# print(f"\n生成的唯一 group_id 数量: {enhanced_df['group_id'].nunique()}")
# print("部分 group_id 示例:")
# print(enhanced_df['group_id'].unique()[:20])

# --- 9. Example of filtering by CF group counts using the new group_id ---
# Example: Find molecules with exactly 6 -CF2- groups
# This can be done by filtering the 'CF2' column directly or by using the group_id string
# subset_cf2_6 = enhanced_df[enhanced_df['CF2'] <= 6]
# print(f"\n找到 {len(subset_cf2_6)} 个包含 6 个 -CF2- 基团的分子 (示例):")
# print(subset_cf2_6[['SMILES', 'first_class', 'second_class', 'CF2', 'CF3', 'CFX', 'group_id']].head())

# Example using group_id string (less efficient, but possible)
# subset_cf2_6_from_gid = enhanced_df[enhanced_df['group_id'].str.contains('CF2_6_')]
# print(f"\n通过 group_id 字符串找到包含 6 个 -CF2- 基团的分子数量: {len(subset_cf2_6_from_gid)}")
