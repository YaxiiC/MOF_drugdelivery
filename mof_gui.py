import os
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)

from dataloader import smiles_to_data
from model import GINToxModel
from moffragmentor_break import process_one


TOXICITY_TEXT_MAP = {
    -1: "Safe",
    0: "Toxic",
    1: "Fatal",
    "-1": "Safe",
    "0": "Toxic",
    "1": "Fatal",
}


def toxicity_label_from_key(label_key: Any) -> str:
    try:
        label_int = int(label_key)
    except (TypeError, ValueError):
        label_int = None
    if label_int is not None and label_int in TOXICITY_TEXT_MAP:
        return TOXICITY_TEXT_MAP[label_int]
    return TOXICITY_TEXT_MAP.get(label_key, str(label_key))


def load_tox_model(model_path: str, device: torch.device):
    """
    Loads the GINToxModel and associated metadata from best_model.pt.
    The checkpoint dictionary contains:
        "model_state_dict",
        "num_node_features",
        "num_edge_features",
        "num_classes",
        "hidden_dim",
        "num_layers",
        "dropout".
    If the checkpoint contains a "label_map" dict, load it; otherwise create a reasonable default mapping.
    Return:
        model (in eval mode),
        label_map (dict mapping label string -> class index).
    """

    checkpoint = torch.load(model_path, map_location=device)
    num_node_features = checkpoint.get("num_node_features")
    num_edge_features = checkpoint.get("num_edge_features")
    num_classes = checkpoint.get("num_classes")
    hidden_dim = checkpoint.get("hidden_dim", 128)
    num_layers = checkpoint.get("num_layers", 5)
    dropout = checkpoint.get("dropout", 0.2)

    model = GINToxModel(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    label_map = checkpoint.get("label_map")
    if label_map is None:
        if num_classes == 3:
            label_map = {"-1": 0, "0": 1, "1": 2}
        else:
            label_map = {str(i): i for i in range(num_classes)}

    return model, label_map


def predict_toxicity_for_smiles_list(
    smiles_list: List[str],
    model: GINToxModel,
    label_map: Dict[str, int],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """
    For each SMILES, build the graph Data object as done during training, run the model, and return:
        {
          "smiles": <str>,
          "pred_class_index": <int or None>,
          "pred_class_label": <str>,
          "probabilities": List[float],
          "status": "ok" or "failed: <error>"
        }
    """

    inv_label_map = {v: k for k, v in label_map.items()}
    results: List[Dict[str, Any]] = []

    for smiles in smiles_list:
        try:
            data = smiles_to_data(smiles, label=0)
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
            data = data.to(device)

            with torch.no_grad():
                logits = model(data.x, data.edge_index, data.edge_attr, data.batch)
                probs_tensor = F.softmax(logits, dim=1)
                probs = probs_tensor.cpu().numpy()[0].tolist()
                pred_idx = int(torch.argmax(logits, dim=1).cpu().item())
                raw_label = inv_label_map.get(pred_idx, pred_idx)
                pred_label = toxicity_label_from_key(raw_label)

            results.append(
                {
                    "smiles": smiles,
                    "pred_class_index": pred_idx,
                    "pred_class_label": pred_label,
                    "raw_label": raw_label,
                    "probabilities": probs,
                    "probability_labels": [
                        toxicity_label_from_key(inv_label_map.get(i, i))
                        for i in range(len(probs))
                    ],
                    "status": "ok",
                }
            )
        except Exception as e:  # pylint: disable=broad-except
            results.append(
                {
                    "smiles": smiles,
                    "pred_class_index": None,
                    "pred_class_label": "failed",
                    "probabilities": [],
                    "status": f"failed: {e}",
                }
            )

    return results


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MOF Linker Toxicity Predictor")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[GINToxModel] = None
        self.label_map: Optional[Dict[str, int]] = None

        self.selected_file: Optional[str] = None

        self._build_ui()
        self._apply_styles()
        self.resize(1000, 700)

    def _build_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout()

        # Input mode tabs
        self.mode_tabs = QTabWidget()
        self.mode_tabs.currentChanged.connect(self.update_run_button_state)

        # CIF file tab
        file_tab = QWidget()
        file_layout = QHBoxLayout()
        self.select_btn = QPushButton("Select CIF File")
        self.select_btn.clicked.connect(self.select_file)
        file_layout.addWidget(self.select_btn)

        self.file_label = QLabel("No file selected")
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        file_layout.addWidget(self.file_label, stretch=1)

        file_tab.setLayout(file_layout)
        self.mode_tabs.addTab(file_tab, "CIF Input")

        # Manual input tab
        manual_tab = QWidget()
        manual_layout = QVBoxLayout()
        manual_row = QHBoxLayout()
        manual_row.addWidget(QLabel("Metals (comma-separated):"))
        self.manual_metals_input = QLineEdit()
        self.manual_metals_input.setPlaceholderText("e.g., Zr, Cu")
        manual_row.addWidget(self.manual_metals_input, stretch=1)
        manual_layout.addLayout(manual_row)

        linker_row = QVBoxLayout()
        linker_row.addWidget(QLabel("Linker SMILES (one per line):"))
        self.manual_linkers_input = QPlainTextEdit()
        self.manual_linkers_input.setPlaceholderText("CCO\nO=C(O)C(=O)O")
        linker_row.addWidget(self.manual_linkers_input)
        manual_layout.addLayout(linker_row)
        manual_tab.setLayout(manual_layout)
        self.mode_tabs.addTab(manual_tab, "Manual Input")

        main_layout.addWidget(self.mode_tabs)

        # Run button row
        run_layout = QHBoxLayout()
        run_layout.addStretch()

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_analysis)
        run_layout.addWidget(self.reset_btn)

        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_analysis)
        run_layout.addWidget(self.run_btn)
        main_layout.addLayout(run_layout)

        # Status labels row
        status_layout = QHBoxLayout()
        self.metals_label = QLabel("Metals detected: -")
        self.linkers_label = QLabel("Number of linkers: 0")
        self.status_label = QLabel("Status: idle")
        status_layout.addWidget(self.metals_label)
        status_layout.addWidget(self.linkers_label)
        status_layout.addWidget(self.status_label)
        main_layout.addLayout(status_layout)

        # Metal toxicity info row
        self.metal_toxicity_label = QLabel("Metal toxicity info: -")
        self.metal_toxicity_label.setObjectName("metalSummary")
        self.metal_toxicity_label.setWordWrap(True)
        main_layout.addWidget(self.metal_toxicity_label)

        # Linker toxicity summary row
        self.linker_summary_label = QLabel(
            "Linker toxicity summary: Safe 0 | Toxic 0 | Fatal 0"
        )
        self.linker_summary_label.setObjectName("linkerSummary")
        self.linker_summary_label.setWordWrap(True)
        main_layout.addWidget(self.linker_summary_label)

        # Table for predictions
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            "#",
            "SMILES",
            "Predicted Toxicity",
            "Class Probabilities",
        ])
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        main_layout.addWidget(self.table)

        # Log area
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        main_layout.addWidget(self.log_widget)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def _apply_styles(self):
        palette_color = "#ffe6f0"
        accent_color = "#d6336c"
        text_color = "#3a2a2a"
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background-color: {palette_color};
                color: {text_color};
            }}
            QLabel {{
                color: {text_color};
                font-size: 14px;
            }}
            QPushButton {{
                background-color: white;
                color: {text_color};
                border: 1px solid {accent_color};
                border-radius: 6px;
                padding: 6px 10px;
                font-weight: bold;
            }}
            QPushButton:disabled {{
                background-color: #f6f6f6;
                color: #9a9a9a;
                border: 1px solid #cccccc;
            }}
            QPushButton:hover:!disabled {{
                background-color: {accent_color};
                color: white;
            }}
            QTableWidget {{
                background: white;
                alternate-background-color: #fce4ec;
                gridline-color: {accent_color};
                selection-background-color: #f8bbd0;
                selection-color: {text_color};
            }}
            QHeaderView::section {{
                background: {accent_color};
                color: white;
                padding: 6px;
                border: none;
            }}
            QTextEdit {{
                background: white;
                border: 1px solid {accent_color};
                border-radius: 6px;
            }}
            QLabel#metalSummary, QLabel#linkerSummary {{
                background: #f8c2d3;
                border: 1px solid {accent_color};
                border-radius: 8px;
                padding: 8px 10px;
                font-weight: bold;
            }}
            """
        )

    def append_log(self, msg: str):
        self.log_widget.append(msg)
        self.log_widget.moveCursor(self.log_widget.textCursor().End)
        self.log_widget.ensureCursorVisible()

    def update_run_button_state(self):
        if not hasattr(self, "run_btn") or self.run_btn is None:
            return
        if self.mode_tabs.currentIndex() == 0:
            self.run_btn.setEnabled(bool(self.selected_file))
        else:
            self.run_btn.setEnabled(True)

    def reset_analysis(self):
        """Clear current selections, results, and logs to start fresh."""
        self.append_log("Resetting analysis state...")
        self.selected_file = None
        self.file_label.setText("No file selected")
        self.manual_metals_input.clear()
        self.manual_linkers_input.clear()

        self.metals_label.setText("Metals detected: -")
        self.linkers_label.setText("Number of linkers: 0")
        self.status_label.setText("Status: idle")
        self.metal_toxicity_label.setText("Metal toxicity info: -")
        self.linker_summary_label.setText(
            "Linker toxicity summary: Safe 0 | Toxic 0 | Fatal 0"
        )

        self.table.clearContents()
        self.table.setRowCount(0)

        self.log_widget.clear()
        self.append_log("Ready for new analysis.")
        self.update_run_button_state()

    def select_file(self):
        self.append_log("Opening file dialog for CIF selection...")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CIF File", "", "CIF Files (*.cif);;All Files (*)"
        )
        if not path:
            self.append_log("File selection cancelled.")
            self.update_run_button_state()
            return

        self.selected_file = path
        self.file_label.setText(path)
        self.append_log(f"Selected CIF: {path}")
        self.update_run_button_state()

    def run_analysis(self):
        use_cif_mode = self.mode_tabs.currentIndex() == 0
        if use_cif_mode and (not self.selected_file or not os.path.isfile(self.selected_file)):
            QMessageBox.warning(self, "No file", "Please select a valid CIF file first.")
            return

        self.append_log("Starting analysis...")
        self.status_label.setText("Status: running")

        if use_cif_mode:
            cif_path = self.selected_file  # type: ignore[arg-type]
            try:
                self.append_log("Parsing CIF and fragmenting...")
                metals, smiles_list, status = process_one(cif_path)
                self.status_label.setText(f"Status: {status}")
                metals_text = ", ".join(metals) if metals else "none"
                self.metals_label.setText(f"Metals detected: {metals_text}")
                self.linkers_label.setText(f"Number of linkers: {len(smiles_list)}")

                self.append_log("Fragmentation completed.")
                self.append_log(f"Found metals: {metals_text}")
                self.append_log(f"Found {len(smiles_list)} linkers.")

                if status.startswith("error"):
                    QMessageBox.critical(self, "Fragmentation Error", status)
                    self.append_log(f"Error during fragmentation: {status}")
                    return

                if not smiles_list:
                    QMessageBox.warning(
                        self,
                        "No Linkers",
                        "No linker SMILES were detected in the provided CIF.",
                    )
                    self.append_log("No linker SMILES detected; aborting analysis.")
                    return

            except Exception as e:  # pylint: disable=broad-except
                err_msg = f"Fragmentation failed: {e}"
                self.status_label.setText(f"Status: error: {e}")
                self.append_log(err_msg)
                QMessageBox.critical(self, "Error", err_msg)
                return
        else:
            self.append_log("Using manual input for metals and linkers.")
            metals_text = self.manual_metals_input.text()
            metals = [m.strip() for m in metals_text.split(",") if m.strip()]
            raw_linkers = self.manual_linkers_input.toPlainText().splitlines()
            smiles_list = [l.strip() for l in raw_linkers if l.strip()]

            self.status_label.setText("Status: manual input")
            metals_display = ", ".join(metals) if metals else "none"
            self.metals_label.setText(f"Metals detected: {metals_display}")
            self.linkers_label.setText(f"Number of linkers: {len(smiles_list)}")

            self.append_log(f"Manual metals: {metals_display}")
            self.append_log(f"Manual linkers count: {len(smiles_list)}")

            if not smiles_list:
                QMessageBox.warning(
                    self,
                    "No Linkers",
                    "Please enter at least one linker SMILES to run analysis.",
                )
                self.append_log("No manual linkers provided; aborting analysis.")
                return

        # Load model lazily
        if self.model is None or self.label_map is None:
            try:
                self.append_log("Loading toxicity model...")
                self.model, self.label_map = load_tox_model("best_model.pt", self.device)
                self.append_log("Model loaded.")
            except Exception as e:  # pylint: disable=broad-except
                err_msg = f"Failed to load model: {e}"
                self.append_log(err_msg)
                QMessageBox.critical(self, "Model Error", err_msg)
                return

        # Predict toxicity
        self.append_log("Running toxicity predictions for linkers...")
        results = predict_toxicity_for_smiles_list(
            smiles_list, self.model, self.label_map, self.device
        )
        self.append_log("Prediction completed.")

        self.update_linker_summary(results)
        self.populate_table(results)
        self.lookup_metal_toxicity(metals)
        self.append_log("Analysis finished.")

    def populate_table(self, results: List[Dict[str, Any]]):
        self.table.clearContents()
        self.table.setRowCount(len(results))

        for row_idx, res in enumerate(results):
            idx_item = QTableWidgetItem(str(row_idx + 1))
            idx_item.setFlags(idx_item.flags() ^ Qt.ItemIsEditable)
            smiles_item = QTableWidgetItem(res.get("smiles", ""))
            smiles_item.setFlags(smiles_item.flags() ^ Qt.ItemIsEditable)
            pred_label = res.get("pred_class_label", "")
            status = res.get("status", "ok")
            if status.startswith("failed"):
                pred_label = "failed"
            pred_item = QTableWidgetItem(str(pred_label))
            pred_item.setFlags(pred_item.flags() ^ Qt.ItemIsEditable)

            probs = res.get("probabilities", [])
            prob_labels = res.get("probability_labels", [])
            prob_display_pairs = []
            for i, p in enumerate(probs):
                label = prob_labels[i] if i < len(prob_labels) else toxicity_label_from_key(i)
                prob_display_pairs.append(f"{label}:{p:.3f}")
            prob_str = "; ".join(prob_display_pairs)
            prob_item = QTableWidgetItem(prob_str)
            prob_item.setFlags(prob_item.flags() ^ Qt.ItemIsEditable)

            self.table.setItem(row_idx, 0, idx_item)
            self.table.setItem(row_idx, 1, smiles_item)
            self.table.setItem(row_idx, 2, pred_item)
            self.table.setItem(row_idx, 3, prob_item)

    def update_linker_summary(self, results: List[Dict[str, Any]]):
        safe = toxic = fatal = 0
        for res in results:
            if not isinstance(res, dict):
                continue
            status = res.get("status", "ok")
            if status.startswith("failed"):
                continue
            label = str(res.get("pred_class_label", "")).strip().lower()
            if label == "safe":
                safe += 1
            elif label == "fatal":
                fatal += 1
            elif label == "toxic":
                toxic += 1
        summary_text = (
            f"Linker toxicity summary: Safe {safe} | Toxic {toxic} | Fatal {fatal}"
        )
        self.linker_summary_label.setText(summary_text)
        self.append_log(
            f"Linker summary -> Safe: {safe}, Toxic: {toxic}, Fatal: {fatal}"
        )

    def lookup_metal_toxicity(self, metals: List[str]):
        if not metals:
            self.metal_toxicity_label.setText("Metal toxicity info: none detected")
            self.append_log("No metals detected; skipping metal toxicity lookup.")
            return

        csv_path = "supporting_information_S8.csv"
        self.append_log(f"Reading metal toxicity CSV: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:  # pylint: disable=broad-except
            self.metal_toxicity_label.setText("Metal toxicity info: error reading CSV")
            self.append_log(f"Failed to read CSV: {e}")
            return

        metal_col = None
        tox_col = None
        for col in df.columns:
            if metal_col is None and "metal" in col.lower():
                metal_col = col
            if tox_col is None and "toxicity" in col.lower():
                tox_col = col
        if metal_col is None or tox_col is None:
            self.metal_toxicity_label.setText(
                "Metal toxicity info: columns not found in CSV"
            )
            self.append_log(
                "Could not determine metal or toxicity columns in the CSV; check headers."
            )
            return

        self.append_log(f"Using columns -> metal: '{metal_col}', toxicity: '{tox_col}'")

        info_parts = []
        df_metal_lower = df[metal_col].astype(str).str.lower()
        for metal in metals:
            matches = df[df_metal_lower == metal.lower()]
            if not matches.empty:
                toxicity_values = matches[tox_col].dropna().astype(str).unique()
                toxicity_str = ", ".join(toxicity_values) if len(toxicity_values) > 0 else "-"
                info_parts.append(f"{metal}: {toxicity_str}")
                self.append_log(f"Metal {metal} toxicity: {toxicity_str}")
            else:
                info_parts.append(f"{metal}: not found")
                self.append_log(f"Metal {metal} not found in toxicity CSV.")

        info_text = "; ".join(info_parts)
        self.metal_toxicity_label.setText(f"Metal toxicity info: {info_text}")


def main():
    import sys

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
