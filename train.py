# train.py

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from dataloader import create_dataset, split_dataset
from model import GINToxModel


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=1).detach().cpu().numpy()
        targets = batch.y.view(-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")

    return avg_loss, acc, f1


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    """
    Evaluate on a loader and return:
      avg_loss, acc, macro-F1, y_true, y_pred, y_prob
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    for batch in loader:
        batch = batch.to(device)

        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.y.view(-1))

        total_loss += loss.item() * batch.num_graphs
        probs = torch.softmax(out, dim=1)

        preds = probs.argmax(dim=1).cpu().numpy()
        targets = batch.y.view(-1).cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets)
        all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.concatenate(all_probs, axis=0) if len(all_probs) > 0 else None

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")

    return avg_loss, acc, f1, all_targets, all_preds, all_probs


def print_detailed_metrics(y_true, y_pred, y_prob, num_classes, label_map, split_name="Val"):
    """
    Print:
      - overall acc, macro-F1, macro-AUC
      - per-class: acc, sensitivity, specificity, F1, AUC
    """
    if y_prob is None:
        print(f"[{split_name}] (no probabilities available, skipping detailed metrics)")
        return

    # Overall metrics
    overall_acc = accuracy_score(y_true, y_pred)
    overall_f1 = f1_score(y_true, y_pred, average="macro")

    overall_auc = None
    try:
        if num_classes == 2:
            overall_auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            overall_auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception as e:
        print(f"[{split_name}] Could not compute overall AUC: {e}")

    print(f"\n[{split_name}] 🔢 Summary metrics every 10 epochs")
    if overall_auc is not None:
        print(
            f"  Overall: Acc = {overall_acc:.4f}, "
            f"Macro-F1 = {overall_f1:.4f}, "
            f"Macro-AUC = {overall_auc:.4f}"
        )
    else:
        print(
            f"  Overall: Acc = {overall_acc:.4f}, "
            f"Macro-F1 = {overall_f1:.4f}, "
            f"Macro-AUC = N/A"
        )

    # Per-class metrics
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    inv_label_map = {v: k for k, v in label_map.items()}

    print(f"\n[{split_name}] 📊 Per-class metrics:")
    print("  Class | Acc     Sens     Spec     F1       AUC")
    print("  -----------------------------------------------")

    total = cm.sum()

    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp

        acc_c = (tp + tn) / total if total > 0 else 0.0
        sens_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall
        spec_c = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # binary F1 for this class vs rest
        y_true_bin = (y_true == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)
        f1_c = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        # AUC for this class (one-vs-rest)
        auc_c = None
        try:
            if y_true_bin.max() == 1 and y_true_bin.min() == 0:
                auc_c = roc_auc_score(y_true_bin, y_prob[:, c])
        except Exception:
            auc_c = None

        label_original = inv_label_map.get(c, c)

        if auc_c is not None:
            print(
                f"  {label_original:>5} | {acc_c:7.4f} {sens_c:8.4f} "
                f"{spec_c:8.4f} {f1_c:8.4f} {auc_c:8.4f}"
            )
        else:
            print(
                f"  {label_original:>5} | {acc_c:7.4f} {sens_c:8.4f} "
                f"{spec_c:8.4f} {f1_c:8.4f}     N/A"
            )
    print("")  # blank line


def main():
    parser = argparse.ArgumentParser(description="Train GNN toxicity classifier")

    parser.add_argument(
        "--csv_path",
        type=str,
        default="/Users/chrissychen/Documents/PhD_2nd_year/MOF_MATRIX/BioMOF_pipeline/oral_data_cleaned.csv",
        help="Path to CSV file with SMILES and labels",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--save_path", type=str, default="best_model.pt")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Create dataset & loaders ---
    data_list, labels, num_node_features, num_edge_features, num_classes, label_map = create_dataset(
        args.csv_path
    )

    print(f"Num samples: {len(data_list)}")
    print(f"Node feature dim: {num_node_features}")
    print(f"Edge feature dim: {num_edge_features}")
    print(f"Num classes: {num_classes}")
    print(f"Label mapping (original -> index): {label_map}")

    train_loader, val_loader, test_loader = split_dataset(
        data_list,
        labels,
        batch_size=args.batch_size,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # --- Model, loss, optimizer ---
    model = GINToxModel(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # === Class weights for imbalance ===
    class_counts = torch.bincount(labels)               # counts per class (0,1,2)
    class_weights = 1.0 / class_counts.float()          # inverse-frequency
    class_weights = class_weights / class_weights.sum() # normalize (optional)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # ===================================

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Training loop ---
    best_val_f1 = 0.0
    patience = 50              # early stopping patience (epochs)
    epochs_no_improve = 0      # counter

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        (
            val_loss,
            val_acc,
            val_f1,
            val_targets,
            val_preds,
            val_probs,
        ) = eval_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}, F1: {val_f1:.4f}"
        )

        # Every 10 epochs: detailed metrics on val set
        if epoch % 10 == 0:
            print_detailed_metrics(
                val_targets,
                val_preds,
                val_probs,
                num_classes=num_classes,
                label_map=label_map,
                split_name="Val",
            )

        # Save best model by validation F1 + early stopping tracking
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0  # reset patience counter
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_node_features": num_node_features,
                    "num_edge_features": num_edge_features,
                    "num_classes": num_classes,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "label_map": label_map,
                },
                args.save_path,
            )
            print(f"  --> New best model saved to {args.save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"\n🛑 Early stopping triggered: no improvement in val F1 for {patience} epochs."
                )
                break

    print("Training finished.")
    print(f"Best validation macro-F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()