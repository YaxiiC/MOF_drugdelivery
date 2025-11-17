#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import warnings

import numpy as np
import pandas as pd
import deepchem as dc
import torch

from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure
from deepchem.splits import RandomSplitter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"Using torch version: {torch.__version__}")

# =========================
# 1. 数据路径 & 读取 CSV
# =========================
csv_path = "C:/Users/chris/MOF_drugdelivery/MOF_drugdelivery/Filtered_Dataset.csv"
properties_df = pd.read_csv(csv_path)
print(f"Total rows in CSV: {len(properties_df)}")

# 取前 6000 个（你原来的设定）
mof_files = properties_df["Filename"].values[:6000]
target_column = "ASA_A^2"
target_values = properties_df[target_column].values[:6000]  # shape (N,)

# CIF 文件夹
cif_directory = "C:/Users/chris/MOF_drugdelivery/2022_CSD_MOF_Collection"

# =========================
# 2. 从 CIF 读取结构（pymatgen）
# =========================
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
        # 新 API: parse_structures 代替 get_structures
        struct_list = parser.parse_structures(primitive=False)
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
with open("rf_skipped_cifs.log", "w") as f:
    for cif_id, reason in skipped:
        f.write(f"{cif_id}\t{reason}\n")

# =========================
# 3. 结构/组分特征（19 维，不用 DGL/CGCNN）
# =========================
# 特征列表：
# 1.  num_atoms     单胞原子数
# 2.  volume        体积
# 3.  density       密度
# 4.  n_elements    元素种类数
# 5.  a, b, c       晶格常数
# 6.  alpha,beta,gamma 晶格角
# 7.  avg_Z, std_Z, min_Z, max_Z   原子序数统计
# 8.  avg_X, std_X                 电负性统计
# 9.  avg_mass                     平均原子质量
# 10. frac_metal, frac_nonmetal    金属/非金属性原子比例

def structure_to_features(struct: Structure) -> np.ndarray:
    comp = struct.composition
    num_atoms = struct.num_sites
    volume = struct.volume
    density = struct.density
    elems = list(comp.elements)
    n_elements = len(elems)

    total = comp.num_atoms
    if total == 0:
        # 避免除零；返回全 0
        return np.zeros(19, dtype=float)

    # Lattice features
    lat = struct.lattice
    a, b, c = lat.a, lat.b, lat.c
    alpha, beta, gamma = lat.alpha, lat.beta, lat.gamma

    # Z, X, mass stats
    sum_Z = 0.0
    sum_Z2 = 0.0
    sum_X = 0.0
    sum_X2 = 0.0
    total_X = 0.0
    sum_mass = 0.0

    min_Z = float("inf")
    max_Z = 0.0

    metal_count = 0.0
    nonmetal_count = 0.0

    for el in elems:
        amt = float(comp[el])  # number of atoms of this element
        Z = el.Z
        sum_Z += Z * amt
        sum_Z2 += (Z ** 2) * amt
        min_Z = min(min_Z, Z)
        max_Z = max(max_Z, Z)

        if el.X is not None:
            sum_X += el.X * amt
            sum_X2 += (el.X ** 2) * amt
            total_X += amt

        if el.atomic_mass is not None:
            sum_mass += float(el.atomic_mass) * amt

        if el.is_metal:
            metal_count += amt
        else:
            nonmetal_count += amt

    avg_Z = sum_Z / total
    var_Z = (sum_Z2 / total) - avg_Z ** 2
    std_Z = max(var_Z, 0.0) ** 0.5

    if total_X > 0:
        avg_X = sum_X / total_X
        var_X = (sum_X2 / total_X) - avg_X ** 2
        std_X = max(var_X, 0.0) ** 0.5
    else:
        avg_X = 0.0
        std_X = 0.0

    avg_mass = sum_mass / total if total > 0 else 0.0
    frac_metal = metal_count / total
    frac_nonmetal = nonmetal_count / total

    return np.array([
        num_atoms,
        volume,
        density,
        n_elements,
        a, b, c,
        alpha, beta, gamma,
        avg_Z, std_Z, min_Z, max_Z,
        avg_X, std_X,
        avg_mass,
        frac_metal,
        frac_nonmetal
    ], dtype=float)

print("Computing handcrafted features...")
X_feats = np.vstack([structure_to_features(s) for s in structures])
y = np.array(labels, dtype=np.float32)

print("Feature matrix shape:", X_feats.shape)
print("Target shape:", y.shape)

# =========================
# 3.5 k-fold 交叉验证（在全部数据上）
# =========================
print("\n===== 5-fold Cross Validation (RandomForest on handcrafted features) =====")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

mae_scores = []
mse_scores = []
rmse_scores = []
r2_scores = []

fold_idx = 1
for train_idx, val_idx in kfold.split(X_feats):
    X_tr, X_va = X_feats[train_idx], X_feats[val_idx]
    y_tr, y_va = y[train_idx], y[val_idx]

    rf_cv = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        random_state=42 + fold_idx,
        n_jobs=-1,
    )
    rf_cv.fit(X_tr, y_tr)
    y_pred = rf_cv.predict(X_va)

    mae = mean_absolute_error(y_va, y_pred)
    mse = mean_squared_error(y_va, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_va, y_pred)

    mae_scores.append(mae)
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    r2_scores.append(r2)

    print(f"[Fold {fold_idx}] MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    fold_idx += 1

def mean_std(arr):
    return float(np.mean(arr)), float(np.std(arr))

mae_mean, mae_std = mean_std(mae_scores)
mse_mean, mse_std = mean_std(mse_scores)
rmse_mean, rmse_std = mean_std(rmse_scores)
r2_mean, r2_std = mean_std(r2_scores)

print("\n[CV 5-fold] Metrics (mean ± std)")
print(f"MAE  = {mae_mean:.4f} ± {mae_std:.4f}")
print(f"MSE  = {mse_mean:.4f} ± {mse_std:.4f}")
print(f"RMSE = {rmse_mean:.4f} ± {rmse_std:.4f}")
print(f"R^2  = {r2_mean:.4f} ± {r2_std:.4f}")

# =========================
# 4. 构建 DeepChem Dataset 并划分（train/valid/test）
# =========================
dataset = dc.data.NumpyDataset(X=X_feats, y=y, ids=np.array(ids))
print("\nDataset size:", len(dataset))

splitter = RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
    dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1
)

print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

# =========================
# 5. 定义非 DGL 模型：RandomForest 回归（DeepChem 包装）
# =========================
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)

model = dc.models.SklearnModel(
    rf,
    model_dir="rf_mof_ASA_model",
    mode="regression"
)

# =========================
# 6. 训练 & 验证
# =========================
metric_mae = dc.metrics.Metric(dc.metrics.mean_absolute_error)
metric_mse = dc.metrics.Metric(dc.metrics.mean_squared_error)
metric_r2 = dc.metrics.Metric(dc.metrics.r2_score)

print("\nFitting RandomForest model on train set...")
model.fit(train_dataset)

print("\nEvaluating on train/valid sets (DeepChem)...")
train_scores = model.evaluate(train_dataset, [metric_mae, metric_mse, metric_r2])
valid_scores = model.evaluate(valid_dataset, [metric_mae, metric_mse, metric_r2])

mae_tr = train_scores["mean_absolute_error"]
mse_tr = train_scores["mean_squared_error"]
rmse_tr = math.sqrt(mse_tr)
r2_tr = train_scores["r2_score"]

mae_va = valid_scores["mean_absolute_error"]
mse_va = valid_scores["mean_squared_error"]
rmse_va = math.sqrt(mse_va)
r2_va = valid_scores["r2_score"]

print(f"[Train] MAE={mae_tr:.4f}, MSE={mse_tr:.4f}, RMSE={rmse_tr:.4f}, R2={r2_tr:.4f}")
print(f"[Valid] MAE={mae_va:.4f}, MSE={mse_va:.4f}, RMSE={rmse_va:.4f}, R2={r2_va:.4f}")

# =========================
# 7. 测试集评估
# =========================
print("\nEvaluating on test set (DeepChem)...")
test_scores = model.evaluate(test_dataset, [metric_mae, metric_mse, metric_r2])
mae_te = test_scores["mean_absolute_error"]
mse_te = test_scores["mean_squared_error"]
rmse_te = math.sqrt(mse_te)
r2_te = test_scores["r2_score"]

print("\n===== Final Test Performance (ASA, RandomForest) =====")
print(f"MAE  = {mae_te:.4f}")
print(f"MSE  = {mse_te:.4f}")
print(f"RMSE = {rmse_te:.4f}")
print(f"R^2  = {r2_te:.4f}")

# =========================
# 8. 保存模型（修正：SklearnModel 用 save() 而不是 save_checkpoint）
# =========================
model.save()
print("RandomForest model saved in folder: rf_mof_ASA_model")
