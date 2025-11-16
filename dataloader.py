# dataloader.py

import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit import RDLogger  # ← add this

RDLogger.DisableLog("rdApp.*") 
from sklearn.model_selection import train_test_split


# --- Atom & bond featurization utils ---

ALLOWED_HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

ALLOWED_BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]

# bond feature size = one-hot bond types + conjugation + ring
BOND_FEATURE_DIM = len(ALLOWED_BOND_TYPES) + 2


def atom_to_feature_vector(atom: rdchem.Atom) -> List[float]:
    """Create a numeric feature vector for a single atom."""
    atomic_num = atom.GetAtomicNum()
    formal_charge = atom.GetFormalCharge()
    implicit_valence = atom.GetImplicitValence()
    explicit_valence = atom.GetExplicitValence()
    is_aromatic = int(atom.GetIsAromatic())
    is_in_ring = int(atom.IsInRing())
    hybridization = atom.GetHybridization()

    # One-hot for hybridization
    hybrid_one_hot = [int(hybridization == h) for h in ALLOWED_HYBRIDIZATIONS]

    features = [
        float(atomic_num),
        float(formal_charge),
        float(implicit_valence),
        float(explicit_valence),
        float(is_aromatic),
        float(is_in_ring),
    ] + hybrid_one_hot

    return features


def bond_to_feature_vector(bond: rdchem.Bond) -> List[float]:
    """Create a numeric feature vector for a single bond."""
    bond_type = bond.GetBondType()
    bond_type_one_hot = [int(bond_type == bt) for bt in ALLOWED_BOND_TYPES]
    is_conjugated = int(bond.GetIsConjugated())
    is_in_ring = int(bond.IsInRing())

    features = bond_type_one_hot + [float(is_conjugated), float(is_in_ring)]
    return features


# --- SMILES → PyG Data ---

def smiles_to_data(smiles: str, label: int) -> Data:
    """Convert a SMILES string + label into a torch_geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")

    # Node features
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(atom_to_feature_vector(atom))

    x = torch.tensor(node_features, dtype=torch.float)

    # Edge index / edge features
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bf = bond_to_feature_vector(bond)

        # Undirected graph: add both directions
        edge_index.append([i, j])
        edge_index.append([j, i])

        edge_attr.append(bf)
        edge_attr.append(bf)

    if len(edge_index) == 0:
        # Handle molecules with no bonds (rare but possible)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    y = torch.tensor([label], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data


# --- Dataset creation & splitting ---

def load_csv(csv_path: str) -> pd.DataFrame:
    """Robust CSV loader (tries comma then tab)."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, sep="\t")
    return df


def create_dataset(
    csv_path: str,
    smiles_column: str = "Canonical SMILES",
    label_column: str = "Category",
) -> Tuple[List[Data], torch.Tensor, int, int, int, Dict[int, int]]:
    """
    Create list of PyG Data objects from CSV.

    Returns:
        data_list: list of Data graphs
        labels: tensor of labels (aligned with data_list)
        num_node_features
        num_edge_features
        num_classes
        label_map: mapping from original label (e.g., -1,0,1) to [0..C-1]
    """
    print(f"\n📄 Loading CSV: {csv_path}")
    df = load_csv(csv_path)
    print(f"➡️ Raw rows: {len(df)}")

    df = df.dropna(subset=[smiles_column, label_column])
    print(f"➡️ After dropping NaN: {len(df)}")

    original_labels = df[label_column].values
    unique_labels = sorted(set(original_labels))
    label_map = {orig: idx for idx, orig in enumerate(unique_labels)}
    print(f"➡️ Label mapping: {label_map}")

    data_list: List[Data] = []
    mapped_labels: List[int] = []

    for idx, row in df.iterrows():
        smiles = row[smiles_column]
        orig_label = row[label_column]
        label = label_map[orig_label]

        try:
            data = smiles_to_data(smiles, label)
        except Exception as e:
            # Skip invalid molecules (you can log if needed)
            print(f"⚠️ Skipping row {idx} | SMILES: {smiles} | Error: {e}")
            continue

        data_list.append(data)
        mapped_labels.append(label)

    print(f"\n✅ Successfully parsed molecules: {len(data_list)}")

    labels = torch.tensor(mapped_labels, dtype=torch.long)

    if len(data_list) == 0:
        raise RuntimeError("No valid molecules could be parsed from the CSV.")

    num_node_features = data_list[0].x.size(1)
    num_edge_features = (
        data_list[0].edge_attr.size(1)
        if data_list[0].edge_attr is not None and data_list[0].edge_attr.numel() > 0
        else 0
    )
    num_classes = len(unique_labels)

    print(f"🧬 Node feature dim: {num_node_features}")
    print(f"🔗 Edge feature dim: {num_edge_features}")
    print(f"🏷️ Num classes: {num_classes}")

    # Print example data
    print("\n🔍 Example graph:")
    print(data_list[0])
    print(f"x shape = {data_list[0].x.shape}")
    print(f"edge_index shape = {data_list[0].edge_index.shape}")
    print(f"edge_attr shape = {data_list[0].edge_attr.shape}")
    print(f"label = {data_list[0].y}\n")

    return data_list, labels, num_node_features, num_edge_features, num_classes, label_map


def split_dataset(
    data_list: List[Data],
    labels: torch.Tensor,
    batch_size: int = 64,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Stratified split into train/val/test and return PyG DataLoaders.
    """
    print("\n📊 Splitting dataset...")

    labels_np = labels.numpy()
    indices = np.arange(len(labels_np))

    # First: train+val vs test
    idx_trainval, idx_test = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels_np,
    )

    # Then: train vs val
    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_ratio_adjusted,
        random_state=seed,
        stratify=labels_np[idx_trainval],
    )

    # Subset via simple lambdas
    def subset(data_list, idxs):
        return [data_list[i] for i in idxs]

    train_dataset = subset(data_list, idx_train)
    val_dataset = subset(data_list, idx_val)
    test_dataset = subset(data_list, idx_test)

    print(f"➡️ Train: {len(train_dataset)}")
    print(f"➡️ Val:   {len(val_dataset)}")
    print(f"➡️ Test:  {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\n✅ DataLoaders created.")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}\n")

    return train_loader, val_loader, test_loader


# --- Simple test when running this file directly ---

if __name__ == "__main__":
    CSV = "/Users/chrissychen/Documents/PhD_2nd_year/MOF_MATRIX/BioMOF_pipeline/oral_data_cleaned.csv"

    print("\n=== Testing dataloader ===")

    data_list, labels, n_node, n_edge, n_classes, label_map = create_dataset(CSV)

    train_loader, val_loader, test_loader = split_dataset(
        data_list,
        labels,
        batch_size=32,
        test_ratio=0.1,
        val_ratio=0.1,
        seed=42,
    )

    print("\n🎉 Dataloader test complete.")