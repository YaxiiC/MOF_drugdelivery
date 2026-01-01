

import os
import numpy as np
import pandas as pd
import deepchem as dc
from packaging import version

import torch
print(f"Using torch version: {torch.__version__}")


from pymatgen.io.cif import CifParser
from deepchem.feat import CGCNNFeaturizer
from deepchem.splits import RandomSplitter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import dgl


# =========================
# 1. Data paths & CSV loading
# =========================
properties_df = pd.read_csv(
    "C:/Users/chris/MOF_drugdelivery/MOF_drugdelivery/Filtered_Dataset.csv"
)
print(f"Total rows in CSV: {len(properties_df)}")

# Use the first 6000 entries (original setting)
mof_files = properties_df["Filename"].values[:6000]
target_column = "Largest Free Sphere"
target_values = properties_df[target_column].values[:6000]  # shape (N,)

# CIF directory
cif_directory = "C:/Users/chris/MOF_drugdelivery/2022_CSD_MOF_Collection"

# =========================
# 2. Read structures from CIF files (pymatgen)
# =========================
from pymatgen.core.structure import Structure

structures = []
labels = []
ids = []
skipped = []

for cif_id, y in zip(mof_files, target_values):
    cif_path = os.path.join(cif_directory, f"{cif_id}.cif")

    if not os.path.exists(cif_path):
        print(f"[WARN] CIF not found, skip: {cif_path}")
        skipped.append((cif_id, "file_not_found"))
        continue

    try:
        parser = CifParser(cif_path, occupancy_tolerance=100.0)
        struct_list = parser.get_structures(primitive=False)
        if len(struct_list) == 0:
            print(f"[WARN] No structure parsed, skip: {cif_path}")
            skipped.append((cif_id, "no_structure"))
            continue

        struct = struct_list[0]
        structures.append(struct)
        labels.append(float(y))
        ids.append(cif_id)

    except Exception as e:
        print(f"[ERROR] Failed to parse {cif_path}: {e}")
        skipped.append((cif_id, str(e)))

print(f"Parsed structures: {len(structures)}")
print(f"Skipped: {len(skipped)}")

# Optional: save skip information
with open("cgcnn_skipped_cifs.log", "w") as f:
    for cif_id, reason in skipped:
        f.write(f"{cif_id}\t{reason}\n")

# =========================
# 3. Create graph features with CGCNNFeaturizer
# =========================
print("Featurizing structures with CGCNNFeaturizer...")
featurizer = CGCNNFeaturizer()  # Default radius 8 Å, up to 12 neighbors

# Returns a list of GraphData objects
X_graphs = featurizer.featurize(structures)
print(f"Number of featurized crystals: {len(X_graphs)}")

# Convert to numpy arrays (object dtype is fine)
X = np.array(X_graphs, dtype=object)
y = np.array(labels, dtype=np.float32).reshape(-1, 1)  # Single regression task

print("X shape (object array of GraphData):", X.shape)
print("y shape:", y.shape)

# =========================
# 4. Build DeepChem Dataset and split
# =========================
# For ~6000 samples, either NumpyDataset or DiskDataset works
dataset = dc.data.DiskDataset.from_numpy(X=X, y=y, ids=np.array(ids))
print("Dataset size:", len(dataset))

splitter = RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
    dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1
)

print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

# =========================
# 5. Define the CGCNN model
# =========================
# Typical configuration from the DeepChem tutorial
model = dc.models.CGCNNModel(
    mode="regression",
    n_tasks=1,
    batch_size=64,
    learning_rate=8e-4,  # Can be tuned if needed
)

# =========================
# 6. Train & validate
# =========================
import math

metric_mae = dc.metrics.Metric(dc.metrics.mean_absolute_error)
metric_mse = dc.metrics.Metric(dc.metrics.mean_squared_error)
metric_r2 = dc.metrics.Metric(dc.metrics.r2_score)

n_epochs = 50  # Start with fewer epochs to check convergence

for epoch in range(1, n_epochs + 1):
    loss = model.fit(train_dataset, nb_epoch=1)
    if epoch % 5 == 0:
        print(f"\n===== Epoch {epoch}/{n_epochs} | training loss: {loss:.4f} =====")

        # Training split
        train_scores = model.evaluate(train_dataset, [metric_mae, metric_mse, metric_r2])
        mae_tr = train_scores["mean_absolute_error"]
        mse_tr = train_scores["mean_squared_error"]
        rmse_tr = math.sqrt(mse_tr)
        r2_tr = train_scores["r2_score"]

        # Validation split
        valid_scores = model.evaluate(valid_dataset, [metric_mae, metric_mse, metric_r2])
        mae_va = valid_scores["mean_absolute_error"]
        mse_va = valid_scores["mean_squared_error"]
        rmse_va = math.sqrt(mse_va)
        r2_va = valid_scores["r2_score"]

        print(f"[Train] MAE={mae_tr:.4f}, MSE={mse_tr:.4f}, RMSE={rmse_tr:.4f}, R2={r2_tr:.4f}")
print(f"[Valid] MAE={mae_va:.4f}, MSE={mse_va:.4f}, RMSE={rmse_va:.4f}, R2={r2_va:.4f}")

# =========================
# 7. Test evaluation
# =========================
test_scores = model.evaluate(test_dataset, [metric_mae, metric_mse, metric_r2])
mae_te = test_scores["mean_absolute_error"]
mse_te = test_scores["mean_squared_error"]
rmse_te = math.sqrt(mse_te)
r2_te = test_scores["r2_score"]

print("\n===== Final Test Performance (Largest Free Sphere) =====")
print(f"MAE  = {mae_te:.4f}")
print(f"MSE  = {mse_te:.4f}")
print(f"RMSE = {rmse_te:.4f}")
print(f"R^2  = {r2_te:.4f}")

# =========================
# 8. Save model (optional)
# =========================
model.save_checkpoint(model_dir="cgcnn_mof_lfs_model")
print("Model saved in folder: cgcnn_mof_lfs_model")
