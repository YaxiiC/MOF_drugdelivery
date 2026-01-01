# fingerprint_baseline.py
"""
Baseline: molecular fingerprints + Random Forest / SVM
Load SMILES and labels from CSV, generate Morgan fingerprints, train, and evaluate models.
Supports multiple random splits (different seeds) and reports mean ± std.
"""

import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

RDLogger.DisableLog("rdApp.*")  # Silence RDKit logs


# ======================
# Data loading & fingerprint construction
# ======================

def load_csv(csv_path: str) -> pd.DataFrame:
    """Robust CSV reader: try comma first, then tab-separated."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, sep="\t")
    return df


def smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """SMILES -> Morgan fingerprint (bit vector)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def create_fingerprint_dataset(
    csv_path: str,
    smiles_column: str = "Canonical SMILES",
    label_column: str = "Category",
    radius: int = 2,
    n_bits: int = 2048,
):
    """
    Build from CSV:
        X: Morgan fingerprints (N, n_bits)
        y: labels (N,), already mapped to [0..C-1]
        label_map: original label -> index
    """
    print(f"\n📄 Reading CSV: {csv_path}")
    df = load_csv(csv_path)
    print(f"➡️ Raw rows: {len(df)}")

    df = df.dropna(subset=[smiles_column, label_column])
    print(f"➡️ After dropping NaN: {len(df)}")

    original_labels = df[label_column].values
    unique_labels = sorted(set(original_labels))
    label_map = {orig: idx for idx, orig in enumerate(unique_labels)}
    print(f"➡️ Label mapping (original -> index): {label_map}")

    fps = []
    mapped_labels = []

    for idx, row in df.iterrows():
        smiles = row[smiles_column]
        orig_label = row[label_column]
        label = label_map[orig_label]
        try:
            fp = smiles_to_morgan_fp(smiles, radius=radius, n_bits=n_bits)
        except Exception as e:
            print(f"⚠️ Skip row {idx}, SMILES={smiles}, error: {e}")
            continue

        fps.append(fp)
        mapped_labels.append(label)

    X = np.vstack(fps).astype(np.float32)
    y = np.array(mapped_labels, dtype=np.int64)

    print(f"\n✅ Fingerprint samples: {X.shape[0]}")
    print(f"🧬 Fingerprint dimension: {X.shape[1]}")
    print(f"🏷️ Number of classes: {len(unique_labels)}")

    return X, y, label_map


def stratified_split(
    X,
    y,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Match the GNN split style: stratified train/val/test split.
    """
    print("\n📊 Splitting dataset (stratified train/val/test)...")

    indices = np.arange(len(y))

    # First split train+val vs test
    idx_trainval, idx_test, y_trainval, y_test = train_test_split(
        indices,
        y,
        test_size=test_ratio,
        random_state=seed,
        stratify=y,
    )

    # Then split val from train+val
    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_trainval,
        y_trainval,
        test_size=val_ratio_adjusted,
        random_state=seed,
        stratify=y_trainval,
    )

    def subset(arr, idxs):
        return arr[idxs]

    X_train, X_val, X_test = subset(X, idx_train), subset(X, idx_val), subset(X, idx_test)
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    print(f"➡️ Train: {len(y_train)}")
    print(f"➡️ Val:   {len(y_val)}")
    print(f"➡️ Test:  {len(y_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ======================
# Metric helpers
# ======================

def compute_per_class_metrics(y_true, y_pred, y_prob, num_classes):
    """
    Return dict: class_idx -> dict(acc, sens, spec, f1, auc)
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    total = cm.sum()
    metrics_per_class = {}

    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp

        acc_c = (tp + tn) / total if total > 0 else 0.0
        sens_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec_c = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # binary f1 (one-vs-rest)
        y_true_bin = (y_true == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)
        f1_c = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        # AUC (one-vs-rest)
        auc_c = None
        try:
            if y_prob is not None:
                auc_c = roc_auc_score(y_true_bin, y_prob[:, c])
        except Exception:
            auc_c = None

        metrics_per_class[c] = {
            "acc": acc_c,
            "sens": sens_c,
            "spec": spec_c,
            "f1": f1_c,
            "auc": auc_c,
        }

    return metrics_per_class


def print_per_class_metrics(y_true, y_pred, y_prob, num_classes, label_map, split_name="Test"):
    metrics_per_class = compute_per_class_metrics(y_true, y_pred, y_prob, num_classes)
    inv_label_map = {v: k for k, v in label_map.items()}

    print(f"\n[{split_name}] 📊 Per-class metrics:")
    print("  Class | Acc     Sens     Spec     F1       AUC")
    print("  -----------------------------------------------")

    for c in range(num_classes):
        m = metrics_per_class[c]
        label_original = inv_label_map.get(c, c)
        if m["auc"] is not None:
            print(
                f"  {label_original:>5} | "
                f"{m['acc']:7.4f} {m['sens']:8.4f} {m['spec']:8.4f} "
                f"{m['f1']:8.4f} {m['auc']:8.4f}"
            )
        else:
            print(
                f"  {label_original:>5} | "
                f"{m['acc']:7.4f} {m['sens']:8.4f} {m['spec']:8.4f} "
                f"{m['f1']:8.4f}     N/A"
            )
    print("")


def evaluate_classifier(clf, X, y, num_classes: int, label_map, split_name="Test"):
    """
    Return overall metrics plus per-class metrics (dict) for later aggregation.
    """
    y_pred = clf.predict(X)
    y_prob = None
    if hasattr(clf, "predict_proba"):
        try:
            y_prob = clf.predict_proba(X)
        except Exception:
            y_prob = None

    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro")

    auc = None
    if y_prob is not None:
        try:
            if num_classes == 2:
                auc = roc_auc_score(y, y_prob[:, 1])
            else:
                auc = roc_auc_score(y, y_prob, multi_class="ovr")
        except Exception:
            auc = None

    print(f"\n=== {split_name} Summary ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    if auc is not None:
        print(f"AUC      : {auc:.4f}")
    else:
        print("AUC      : N/A")

    # Classification report
    inv_label_map = {v: k for k, v in label_map.items()}
    target_names = [str(inv_label_map[i]) for i in range(num_classes)]
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=target_names))

    # per-class metrics (print)
    print_per_class_metrics(y, y_pred, y_prob, num_classes, label_map, split_name)

    # per-class metrics (return for mean ± std)
    per_class = compute_per_class_metrics(y, y_pred, y_prob, num_classes)

    return acc, macro_f1, auc, per_class


# ======================
# Main: multiple random splits + RF / SVM
# ======================

def main():
    parser = argparse.ArgumentParser(description="Fingerprint + RF / SVM baseline with multi-run stats")

    parser.add_argument(
        "--csv_path",
        type=str,
        default="/Users/chrissychen/Documents/PhD_2nd_year/MOF_MATRIX/BioMOF_pipeline/oral_data_cleaned.csv",
        help="CSV path (contains SMILES and Category columns)",
    )
    parser.add_argument("--smiles_column", type=str, default="Canonical SMILES")
    parser.add_argument("--label_column", type=str, default="Category")
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius")
    parser.add_argument("--n_bits", type=int, default=2048, help="Morgan fingerprint length")
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs with different random seeds")

    # RF hyperparameters
    parser.add_argument("--rf_n_estimators", type=int, default=500)
    parser.add_argument("--rf_max_depth", type=int, default=None)

    # SVM hyperparameters
    parser.add_argument("--svm_C", type=float, default=1.0)
    parser.add_argument("--svm_kernel", type=str, default="rbf")  # 'linear', 'rbf', 'poly', ...

    args = parser.parse_args()

    # 1. Build fingerprint dataset
    X, y, label_map = create_fingerprint_dataset(
        args.csv_path,
        smiles_column=args.smiles_column,
        label_column=args.label_column,
        radius=args.radius,
        n_bits=args.n_bits,
    )
    num_classes = len(set(y))

    # =======================
    #  Random Forest: multiple runs
    # =======================
    rf_accs, rf_f1s, rf_aucs = [], [], []
    rf_per_class_stats = {
        c: {"acc": [], "sens": [], "spec": [], "f1": [], "auc": []}
        for c in range(num_classes)
    }

    print("\n" + "=" * 70)
    print(f"🌲 Random Forest: {args.num_runs} runs with seeds {args.seed}..{args.seed + args.num_runs - 1}")

    for i in range(args.num_runs):
        seed_i = args.seed + i
        print(f"\n--- RF Run {i+1}/{args.num_runs} (seed={seed_i}) ---")

        X_train, y_train, X_val, y_val, X_test, y_test = stratified_split(
            X,
            y,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            seed=seed_i,
        )

        rf = RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            random_state=seed_i,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        rf.fit(X_train, y_train)

        # Focus on Test metrics (aligned with GNN evaluation)
        acc, macro_f1, auc, per_class = evaluate_classifier(
            rf, X_test, y_test, num_classes, label_map, split_name="RF Test"
        )

        rf_accs.append(acc)
        rf_f1s.append(macro_f1)
        rf_aucs.append(auc if auc is not None else np.nan)

        # Aggregate per-class stats
        for c in range(num_classes):
            rf_per_class_stats[c]["acc"].append(per_class[c]["acc"])
            rf_per_class_stats[c]["sens"].append(per_class[c]["sens"])
            rf_per_class_stats[c]["spec"].append(per_class[c]["spec"])
            rf_per_class_stats[c]["f1"].append(per_class[c]["f1"])
            rf_per_class_stats[c]["auc"].append(
                per_class[c]["auc"] if per_class[c]["auc"] is not None else np.nan
            )

    rf_accs = np.array(rf_accs)
    rf_f1s = np.array(rf_f1s)
    rf_aucs = np.array(rf_aucs)

    print("\n=== RF Test performance over runs (mean ± std) ===")
    print(f"Accuracy: {rf_accs.mean():.4f} ± {rf_accs.std():.4f}")
    print(f"Macro-F1: {rf_f1s.mean():.4f} ± {rf_f1s.std():.4f}")
    if np.isfinite(rf_aucs).any():
        print(f"AUC:      {np.nanmean(rf_aucs):.4f} ± {np.nanstd(rf_aucs):.4f}")
    else:
        print("AUC:      N/A")

    inv_label_map = {v: k for k, v in label_map.items()}
    print("\n=== RF per-class Test performance over runs (mean ± std) ===")
    print("Class |   Acc (µ±σ)       Sens (µ±σ)      Spec (µ±σ)      F1 (µ±σ)        AUC (µ±σ)")
    print("-------------------------------------------------------------------------------------")
    for c in range(num_classes):
        stats = rf_per_class_stats[c]
        acc_arr = np.array(stats["acc"])
        sens_arr = np.array(stats["sens"])
        spec_arr = np.array(stats["spec"])
        f1_arr = np.array(stats["f1"])
        auc_arr = np.array(stats["auc"])

        def fmt(arr):
            if np.isfinite(arr).any():
                return f"{np.nanmean(arr):.4f} ± {np.nanstd(arr):.4f}"
            else:
                return "N/A"

        label_original = inv_label_map.get(c, c)
        print(
            f"{label_original:>5} | {fmt(acc_arr):>15}  {fmt(sens_arr):>15}  "
            f"{fmt(spec_arr):>15}  {fmt(f1_arr):>15}  {fmt(auc_arr):>15}"
        )

    # =======================
    #  SVM: multiple runs
    # =======================
    svm_accs, svm_f1s, svm_aucs = [], [], []
    svm_per_class_stats = {
        c: {"acc": [], "sens": [], "spec": [], "f1": [], "auc": []}
        for c in range(num_classes)
    }

    print("\n" + "=" * 70)
    print(f"🧠 SVM: {args.num_runs} runs with seeds {args.seed}..{args.seed + args.num_runs - 1}")
    print(f"    kernel={args.svm_kernel}, C={args.svm_C}")

    for i in range(args.num_runs):
        seed_i = args.seed + i
        print(f"\n--- SVM Run {i+1}/{args.num_runs} (seed={seed_i}) ---")

        X_train, y_train, X_val, y_val, X_test, y_test = stratified_split(
            X,
            y,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            seed=seed_i,
        )

        svm = SVC(
            C=args.svm_C,
            kernel=args.svm_kernel,
            gamma="scale",
            probability=True,   # For AUC and per-class probabilities
            random_state=seed_i,
        )
        svm.fit(X_train, y_train)

        acc, macro_f1, auc, per_class = evaluate_classifier(
            svm, X_test, y_test, num_classes, label_map, split_name="SVM Test"
        )

        svm_accs.append(acc)
        svm_f1s.append(macro_f1)
        svm_aucs.append(auc if auc is not None else np.nan)

        for c in range(num_classes):
            svm_per_class_stats[c]["acc"].append(per_class[c]["acc"])
            svm_per_class_stats[c]["sens"].append(per_class[c]["sens"])
            svm_per_class_stats[c]["spec"].append(per_class[c]["spec"])
            svm_per_class_stats[c]["f1"].append(per_class[c]["f1"])
            svm_per_class_stats[c]["auc"].append(
                per_class[c]["auc"] if per_class[c]["auc"] is not None else np.nan
            )

    svm_accs = np.array(svm_accs)
    svm_f1s = np.array(svm_f1s)
    svm_aucs = np.array(svm_aucs)

    print("\n=== SVM Test performance over runs (mean ± std) ===")
    print(f"Accuracy: {svm_accs.mean():.4f} ± {svm_accs.std():.4f}")
    print(f"Macro-F1: {svm_f1s.mean():.4f} ± {svm_f1s.std():.4f}")
    if np.isfinite(svm_aucs).any():
        print(f"AUC:      {np.nanmean(svm_aucs):.4f} ± {np.nanstd(svm_aucs):.4f}")
    else:
        print("AUC:      N/A")

    print("\n=== SVM per-class Test performance over runs (mean ± std) ===")
    print("Class |   Acc (µ±σ)       Sens (µ±σ)      Spec (µ±σ)      F1 (µ±σ)        AUC (µ±σ)")
    print("-------------------------------------------------------------------------------------")
    for c in range(num_classes):
        stats = svm_per_class_stats[c]
        acc_arr = np.array(stats["acc"])
        sens_arr = np.array(stats["sens"])
        spec_arr = np.array(stats["spec"])
        f1_arr = np.array(stats["f1"])
        auc_arr = np.array(stats["auc"])

        def fmt(arr):
            if np.isfinite(arr).any():
                return f"{np.nanmean(arr):.4f} ± {np.nanstd(arr):.4f}"
            else:
                return "N/A"

        label_original = inv_label_map.get(c, c)
        print(
            f"{label_original:>5} | {fmt(acc_arr):>15}  {fmt(sens_arr):>15}  "
            f"{fmt(spec_arr):>15}  {fmt(f1_arr):>15}  {fmt(auc_arr):>15}"
        )

    print("\n✅ Fingerprint RF + SVM baseline multi-run experiment completed.")


if __name__ == "__main__":
    main()
