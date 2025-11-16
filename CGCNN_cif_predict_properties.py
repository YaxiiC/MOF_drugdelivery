import os
import numpy as np
import pandas as pd
import deepchem as dc

from pymatgen.io.cif import CifParser
from deepchem.feat import CGCNNFeaturizer
from deepchem.splits import RandomSplitter

# =========================
# 1. 数据路径 & 读取 CSV
# =========================
properties_df = pd.read_csv(
    "/Users/chrissychen/Documents/PhD_2nd_year/MOF_MATRIX/zeyi_msc_project/Filtered_Dataset.csv"
)
print(f"Total rows in CSV: {len(properties_df)}")

# 取前 6000 个（你原来的设定）
mof_files = properties_df["Filename"].values[:6000]
target_column = "Largest Free Sphere"
target_values = properties_df[target_column].values[:6000]  # shape (N,)

# CIF 文件夹
cif_directory = "/Users/chrissychen/Documents/PhD_2nd_year/MOF_MATRIX/2022_CSD_MOF_Collection"

# =========================
# 2. 从 CIF 读取结构（pymatgen）
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

# 可选：保存一下 skip 的信息
with open("cgcnn_skipped_cifs.log", "w") as f:
    for cif_id, reason in skipped:
        f.write(f"{cif_id}\t{reason}\n")

# =========================
# 3. 用 CGCNNFeaturizer 对结构做图特征
# =========================
print("Featurizing structures with CGCNNFeaturizer...")
featurizer = CGCNNFeaturizer()  # 默认半径 8 Å，最多 12 邻居

# 返回的是 GraphData 对象列表
X_graphs = featurizer.featurize(structures)
print(f"Number of featurized crystals: {len(X_graphs)}")

# 转成 numpy 数组（object 类型即可）
X = np.array(X_graphs, dtype=object)
y = np.array(labels, dtype=np.float32).reshape(-1, 1)  # 1 个回归任务

print("X shape (object array of GraphData):", X.shape)
print("y shape:", y.shape)

# =========================
# 4. 构建 DeepChem Dataset 并划分
# =========================
# 对 6000 左右的数据，用 NumpyDataset 或 DiskDataset 都可以
dataset = dc.data.DiskDataset.from_numpy(X=X, y=y, ids=np.array(ids))
print("Dataset size:", len(dataset))

splitter = RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
    dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1
)

print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

# =========================
# 5. 定义 CGCNN 模型
# =========================
# 这是 DeepChem 官方 tutorial 中的典型配置
model = dc.models.CGCNNModel(
    mode="regression",
    n_tasks=1,
    batch_size=64,
    learning_rate=8e-4,  # 你可以稍微调参
)

# =========================
# 6. 训练 & 验证
# =========================
import math

metric_mae = dc.metrics.Metric(dc.metrics.mean_absolute_error)
metric_mse = dc.metrics.Metric(dc.metrics.mean_squared_error)
metric_r2 = dc.metrics.Metric(dc.metrics.r2_score)

n_epochs = 50  # 可以先跑少一点看收敛情况

for epoch in range(1, n_epochs + 1):
    loss = model.fit(train_dataset, nb_epoch=1)
    if epoch % 5 == 0:
        print(f"\n===== Epoch {epoch}/{n_epochs} | training loss: {loss:.4f} =====")

        # 训练集
        train_scores = model.evaluate(train_dataset, [metric_mae, metric_mse, metric_r2])
        mae_tr = train_scores["mean_absolute_error"]
        mse_tr = train_scores["mean_squared_error"]
        rmse_tr = math.sqrt(mse_tr)
        r2_tr = train_scores["r2_score"]

        # 验证集
        valid_scores = model.evaluate(valid_dataset, [metric_mae, metric_mse, metric_r2])
        mae_va = valid_scores["mean_absolute_error"]
        mse_va = valid_scores["mean_squared_error"]
        rmse_va = math.sqrt(mse_va)
        r2_va = valid_scores["r2_score"]

        print(f"[Train] MAE={mae_tr:.4f}, MSE={mse_tr:.4f}, RMSE={rmse_tr:.4f}, R2={r2_tr:.4f}")
        print(f"[Valid] MAE={mae_va:.4f}, MSE={mse_va:.4f}, RMSE={rmse_va:.4f}, R2={r2_va:.4f}")

# =========================
# 7. 测试集评估
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
# 8. 保存模型（可选）
# =========================
model.save_checkpoint(model_dir="cgcnn_mof_lfs_model")
print("Model saved in folder: cgcnn_mof_lfs_model")