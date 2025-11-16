# evaluate.py

import argparse
import os  # NEW

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

import matplotlib.pyplot as plt  # NEW
from torch_geometric.explain import GNNExplainer
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

from dataloader import create_dataset, split_dataset
from model import GINToxModel
import networkx as nx


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs = torch.softmax(out, dim=1)

        preds = probs.argmax(dim=1).cpu().numpy()
        targets = batch.y.view(-1).cpu().numpy()
        probs_np = probs.cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets)
        all_probs.append(probs_np)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.concatenate(all_probs, axis=0) if len(all_probs) > 0 else None

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")

    # AUC (if applicable)
    auc = None
    if all_probs is not None:
        num_classes = all_probs.shape[1]
        try:
            if num_classes == 2:
                auc = roc_auc_score(all_targets, all_probs[:, 1])
            else:
                auc = roc_auc_score(all_targets, all_probs, multi_class="ovr")
        except Exception as e:
            print(f"Could not compute AUC: {e}")

    return acc, f1, auc, all_targets, all_preds, all_probs


def compute_per_class_metrics(y_true, y_pred, y_prob, num_classes):
    """
    Return dict: class_idx -> dict with keys:
      'acc', 'sens', 'spec', 'f1', 'auc'
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
        sens_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall
        spec_c = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # binary F1 for this class vs rest
        y_true_bin = (y_true == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)
        f1_c = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        # AUC for this class (one-vs-rest)
        auc_c = None
        try:
            if y_prob is not None and y_true_bin.max() == 1 and y_true_bin.min() == 0:
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


def print_detailed_metrics(y_true, y_pred, y_prob, num_classes, label_map, split_name="Test"):
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

    print(f"\n[{split_name}] 🔢 Summary metrics")
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
                f"  {label_original:>5} | {m['acc']:7.4f} {m['sens']:8.4f} "
                f"{m['spec']:8.4f} {m['f1']:8.4f} {m['auc']:8.4f}"
            )
        else:
            print(
                f"  {label_original:>5} | {m['acc']:7.4f} {m['sens']:8.4f} "
                f"{m['spec']:8.4f} {m['f1']:8.4f}     N/A"
            )
    print("")  # blank line


def run_gnn_explainer(
    model,
    data_list,
    label_map,
    device,
    example_idx: int = 0,
    out_path: str = "gnn_explainer_example.txt",
):
    """
    Run GNNExplainer (new PyG API) on a single example graph and
    print/save top important nodes/edges + a PNG visualization.
    """
    model.eval()

    if example_idx < 0 or example_idx >= len(data_list):
        example_idx = 0
    data = data_list[example_idx].to(device)

    # Single graph: batch = all zeros
    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    # Configure explainer for graph-level multiclass classification
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="model",
        node_mask_type="attributes",  # per-node *feature* mask
        edge_mask_type="object",
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="graph",
            return_type="probs",
        ),
    )

    # Call explainer on graph
    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=getattr(data, "edge_attr", None),
        batch=batch,
    )

    # node_mask may be (num_nodes, num_features); reduce to scalar per node
    node_mask_raw = explanation.node_mask.detach().cpu().numpy()
    if node_mask_raw.ndim == 2:
        node_importance = node_mask_raw.mean(axis=-1)
    else:
        node_importance = node_mask_raw  # already (num_nodes,)

    edge_mask = explanation.edge_mask.detach().cpu().numpy()

    # Simple text "visualization": top-k important nodes/edges
    k_nodes = min(10, len(node_importance))
    k_edges = min(10, len(edge_mask))

    top_node_idx = node_importance.argsort()[::-1][:k_nodes]
    top_edge_idx = edge_mask.argsort()[::-1][:k_edges]

    lines = []
    lines.append(f"GNNExplainer results for example graph index {example_idx}")
    lines.append("")
    lines.append("Top important nodes:")
    for i in top_node_idx:
        i_int = int(i)
        lines.append(f"  node {i_int} | importance = {node_importance[i_int]:.4f}")

    lines.append("")
    lines.append("Top important edges:")
    for idx in top_edge_idx:
        idx_int = int(idx)
        src = int(data.edge_index[0, idx_int])
        dst = int(data.edge_index[1, idx_int])
        lines.append(
            f"  edge {src} -> {dst} | importance = {edge_mask[idx_int]:.4f}"
        )

    text = "\n".join(lines)
    print("\n=== GNNExplainer (text view) ===")
    print(text)

    # Save text explanation
    txt_path = out_path
    if not txt_path.endswith(".txt"):
        txt_path = txt_path + ".txt"
    with open(txt_path, "w") as f:
        f.write(text)
    print(f"\nGNNExplainer text explanation saved to: {txt_path}")

    # ---- PNG visualization with networkx ----
    # Build an undirected graph with edge importance as weight
    G = nx.Graph()
    num_nodes = data.x.size(0)
    G.add_nodes_from(range(num_nodes))

    num_edges = data.edge_index.size(1)
    for e_idx in range(num_edges):
        u = int(data.edge_index[0, e_idx])
        v = int(data.edge_index[1, e_idx])
        w = float(edge_mask[e_idx])
        if G.has_edge(u, v):
            # keep the max importance if edge appears twice (due to undirected)
            if w > G[u][v].get("weight", 0.0):
                G[u][v]["weight"] = w
        else:
            G.add_edge(u, v, weight=w)

    # Layout for visualization
    pos = nx.spring_layout(G, seed=42)

    node_colors = [float(node_importance[i]) for i in range(num_nodes)]
    edge_widths = [G[u][v]["weight"] * 5.0 for u, v in G.edges()]  # scale for visibility

    plt.figure(figsize=(5, 5))
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        cmap="viridis",
        node_size=200,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        alpha=0.7,
    )
    nx.draw_networkx_labels(G, pos, font_size=6)

    plt.colorbar(nodes, label="Node importance")
    plt.axis("off")
    plt.tight_layout()

    png_base, _ = os.path.splitext(out_path)
    png_path = png_base + ".png"
    plt.savefig(png_path, dpi=300)
    plt.close()

    print(f"GNNExplainer graph visualization saved to: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GNN toxicity classifier")

    parser.add_argument(
        "--csv_path",
        type=str,
        default="/Users/chrissychen/Documents/PhD_2nd_year/MOF_MATRIX/BioMOF_pipeline/oral_data_cleaned.csv",
        help="Path to CSV file with SMILES and labels",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_runs", type=int, default=5)  # for mean±std
    parser.add_argument("--model_path", type=str, default="best_model.pt")
    parser.add_argument("--num_explain", type=int, default=1)  # kept, but we’ll just use 1

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(args.model_path, map_location=device)

    # Load dataset once
    data_list, labels, num_node_features, num_edge_features, num_classes, label_map = create_dataset(
        args.csv_path
    )

    # Build model with same architecture
    model = GINToxModel(
        num_node_features=checkpoint["num_node_features"],
        num_edge_features=checkpoint["num_edge_features"],
        num_classes=checkpoint["num_classes"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    accs = []
    f1s = []
    aucs = []

    # per-class metrics accumulator: class_idx -> metric -> list of values
    per_class_stats = {
        c: {"acc": [], "sens": [], "spec": [], "f1": [], "auc": []}
        for c in range(num_classes)
    }

    last_run_results = None  # (y_true, y_pred, y_prob)

    print(
        f"\nRunning evaluation for {args.num_runs} runs with seeds "
        f"{args.seed} .. {args.seed + args.num_runs - 1}"
    )

    for i in range(args.num_runs):
        seed_i = args.seed + i

        # Re-split dataset with different seed each time
        train_loader, val_loader, test_loader = split_dataset(
            data_list,
            labels,
            batch_size=args.batch_size,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            seed=seed_i,
        )

        acc, f1, auc, y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)

        accs.append(acc)
        f1s.append(f1)
        aucs.append(auc if auc is not None else np.nan)

        # per-class metrics for this run
        metrics_per_class = compute_per_class_metrics(y_true, y_pred, y_prob, num_classes)
        for c in range(num_classes):
            per_class_stats[c]["acc"].append(metrics_per_class[c]["acc"])
            per_class_stats[c]["sens"].append(metrics_per_class[c]["sens"])
            per_class_stats[c]["spec"].append(metrics_per_class[c]["spec"])
            per_class_stats[c]["f1"].append(metrics_per_class[c]["f1"])
            per_class_stats[c]["auc"].append(
                metrics_per_class[c]["auc"] if metrics_per_class[c]["auc"] is not None else np.nan
            )

        print(f"\nRun {i+1}/{args.num_runs} (seed={seed_i})")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Macro-F1: {f1:.4f}")
        if auc is not None:
            print(f"  AUC:      {auc:.4f}")
        else:
            print("  AUC:      N/A")

        # store for detailed metrics (we'll use the last run)
        last_run_results = (y_true, y_pred, y_prob)

    accs = np.array(accs)
    f1s = np.array(f1s)
    aucs = np.array(aucs)

    print("\n=== Test performance over runs (mean ± std) ===")
    print(f"Accuracy: {accs.mean():.4f} ± {accs.std():.4f}")
    print(f"Macro-F1: {f1s.mean():.4f} ± {f1s.std():.4f}")

    # AUC: ignore NaNs if some runs failed to compute it
    if np.isfinite(aucs).any():
        auc_mean = np.nanmean(aucs)
        auc_std = np.nanstd(aucs)
        print(f"AUC:      {auc_mean:.4f} ± {auc_std:.4f}")
    else:
        print("AUC:      N/A")

    # === Per-class mean ± std across runs ===
    inv_label_map = {v: k for k, v in label_map.items()}
    print("\n=== Per-class performance over runs (mean ± std) ===")
    print("Class |   Acc (µ±σ)       Sens (µ±σ)      Spec (µ±σ)      F1 (µ±σ)        AUC (µ±σ)")
    print("-------------------------------------------------------------------------------------")
    for c in range(num_classes):
        stats = per_class_stats[c]
        acc_arr = np.array(stats["acc"])
        sens_arr = np.array(stats["sens"])
        spec_arr = np.array(stats["spec"])
        f1_arr = np.array(stats["f1"])
        auc_arr = np.array(stats["auc"])

        label_original = inv_label_map.get(c, c)

        def fmt(arr):
            if np.isfinite(arr).any():
                return f"{np.nanmean(arr):.4f} ± {np.nanstd(arr):.4f}"
            else:
                return "N/A"

        acc_str = fmt(acc_arr)
        sens_str = fmt(sens_arr)
        spec_str = fmt(spec_arr)
        f1_str = fmt(f1_arr)
        auc_str = fmt(auc_arr)

        print(
            f"{label_original:>5} | {acc_str:>15}  {sens_str:>15}  "
            f"{spec_str:>15}  {f1_str:>15}  {auc_str:>15}"
        )

    # Detailed metrics for the last run
    if last_run_results is not None:
        y_true, y_pred, y_prob = last_run_results
        print_detailed_metrics(
            y_true,
            y_pred,
            y_prob,
            num_classes=num_classes,
            label_map=label_map,
            split_name="Test (last run)",
        )

    # GNNExplainer on a few example graphs (text + PNG)
    num_explain = min(args.num_explain, len(data_list))
    print(f"\nRunning GNNExplainer on {num_explain} example(s)...")

    for ex_idx in range(num_explain):
        base = f"gnn_explainer_example_{ex_idx}"
        run_gnn_explainer(
            model=model,
            data_list=data_list,
            label_map=label_map,
            device=device,
            example_idx=ex_idx,
            out_path=base,  # will create base.txt and base.png
        )


if __name__ == "__main__":
    main()