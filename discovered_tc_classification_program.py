from __future__ import annotations

import csv
import os
from pathlib import Path


def load_classification_rows(csv_path: str | Path, data_root: str | Path) -> list[dict]:
    """Load crystal classification rows and require every CIF to parse."""
    from pymatgen.core import Structure

    csv_path = Path(csv_path)
    data_root = Path(data_root)
    rows: list[dict] = []

    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            label_raw = row.get("label", "")
            if label_raw == "":
                continue

            cif_path = data_root / row["cif"]
            cif_text = cif_path.read_text(errors="replace")
            structure = Structure.from_str(cif_text, fmt="cif")
            rows.append(
                {
                    "formula": row.get("formula", ""),
                    "icsd_collection_code": row.get("icsd_collection_code", ""),
                    "source": row.get("source", ""),
                    "label": label_raw,
                    "chunk": row.get("chunk", ""),
                    "tc": row.get("Tc", ""),
                    "cif_path": str(cif_path),
                    "cif_text": cif_text,
                    "structure": structure,
                }
            )

    return rows


# EVOLVE-BLOCK-START
def fit_predict(train_rows: list[dict], test_rows: list[dict]) -> list:
    """
    Featurizes crystal structures into graphs, trains the discovered GNN backbone
    as a classifier, and predicts labels for the test set.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from pymatgen.core import Element
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATv2Conv, Set2Set
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.deprecation")

    if not train_rows:
        return [None for _ in test_rows]

    labels = [row["label"] for row in train_rows]
    label_encoder = LabelEncoder().fit(labels)
    encoded_labels = label_encoder.transform(labels)
    num_classes = len(label_encoder.classes_)

    if num_classes == 1:
        only_label = label_encoder.classes_[0]
        return [only_label for _ in test_rows]

    # --- Hyperparameters and Setup ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = int(os.environ.get("TC_CLASS_EPOCHS", os.environ.get("TC_EPOCHS", "150")))
    LOG_EVERY = int(os.environ.get("TC_LOG_EVERY", "100"))
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    HIDDEN_DIM = 128
    CUTOFF_RADIUS = 5.0
    NUM_ATOM_FEATURES = 11
    NUM_EXPANDED_EDGE_FEATURES = 32

    # --- Featurization ---
    ELEMENT_FEATURE_VECTORS = {}
    feature_names = [
        "Z",
        "X",
        "atomic_mass",
        "atomic_radius",
        "row",
        "group",
        "electron_affinity",
        "ionization_energy",
        "thermal_conductivity",
        "electrical_resistivity",
        "mendeleev_no",
    ]
    for i in range(1, 104):
        try:
            el = Element.from_Z(i)
            features = [getattr(el, name, 0) for name in feature_names]
            ELEMENT_FEATURE_VECTORS[el.symbol] = np.array(
                [f if f is not None else 0 for f in features], dtype=float
            )
        except (KeyError, ValueError):
            pass

    def get_element_features(element_symbol: str) -> np.ndarray:
        return ELEMENT_FEATURE_VECTORS.get(element_symbol, np.zeros(NUM_ATOM_FEATURES))

    class GaussianExpansion(nn.Module):
        def __init__(self, dmin, dmax, steps):
            super().__init__()
            self.centers = torch.linspace(dmin, dmax, steps)
            self.width = (dmax - dmin) / steps

        def forward(self, distances: torch.Tensor) -> torch.Tensor:
            centers = self.centers.to(distances.device)
            return torch.exp(-0.5 * (((distances - centers) / self.width) ** 2))

    distance_expander = GaussianExpansion(
        dmin=0.0,
        dmax=CUTOFF_RADIUS,
        steps=NUM_EXPANDED_EDGE_FEATURES,
    )

    def structure_to_graph(structure) -> Data:
        atom_features = []
        for site in structure:
            site_features = np.zeros(NUM_ATOM_FEATURES)
            for element, occupancy in site.species.items():
                site_features += occupancy * get_element_features(element.symbol)
            atom_features.append(site_features)
        x = torch.tensor(np.array(atom_features), dtype=torch.float)

        if x.shape[0] == 0:
            raise ValueError("Structure has no atoms and cannot be featurized.")

        edge_src, edge_dst, edge_distances = [], [], []
        neighbor_list = structure.get_all_neighbors(r=CUTOFF_RADIUS)
        for i, neighbors in enumerate(neighbor_list):
            for neighbor in neighbors:
                edge_src.append(i)
                edge_dst.append(neighbor.index)
                edge_distances.append(neighbor.nn_distance)

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        distances = torch.tensor(edge_distances, dtype=torch.float).view(-1, 1)
        edge_attr = distance_expander(distances)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    class GNNClassifier(nn.Module):
        def __init__(self, num_node_features, num_edge_features, hidden_dim, n_classes):
            super().__init__()
            heads = 4
            self.conv1 = GATv2Conv(
                num_node_features,
                hidden_dim,
                heads=heads,
                concat=False,
                edge_dim=num_edge_features,
            )
            self.bn1 = nn.LayerNorm(hidden_dim)

            self.conv2 = GATv2Conv(
                hidden_dim,
                hidden_dim,
                heads=heads,
                concat=False,
                edge_dim=num_edge_features,
            )
            self.bn2 = nn.LayerNorm(hidden_dim)

            self.conv3 = GATv2Conv(
                hidden_dim,
                hidden_dim,
                heads=heads,
                concat=False,
                edge_dim=num_edge_features,
            )
            self.bn3 = nn.LayerNorm(hidden_dim)

            self.set2set = Set2Set(hidden_dim, processing_steps=3)
            self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, n_classes)
            self.dropout = nn.Dropout(0.25)

        def forward(self, data: Data) -> torch.Tensor:
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )

            x = F.elu(self.conv1(x, edge_index, edge_attr))
            x = self.bn1(x)

            x_res = x
            x = F.elu(self.conv2(x, edge_index, edge_attr))
            x = self.bn2(x)
            x = x + x_res

            x_res = x
            x = F.elu(self.conv3(x, edge_index, edge_attr))
            x = self.bn3(x)
            x = x + x_res

            x = self.set2set(x, batch)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)

    print(
        f"[tc-class] device={DEVICE} epochs={EPOCHS} "
        f"train_rows={len(train_rows)} test_rows={len(test_rows)} "
        f"classes={list(label_encoder.classes_)}",
        flush=True,
    )
    print("[tc-class] featurizing structures into graphs...", flush=True)
    train_graphs = [structure_to_graph(row["structure"]) for row in train_rows]
    test_graphs = [structure_to_graph(row["structure"]) for row in test_rows]
    print(
        f"[tc-class] graph featurization complete: "
        f"train_graphs={len(train_graphs)} test_graphs={len(test_graphs)}",
        flush=True,
    )

    all_train_node_features = np.vstack([g.x.numpy() for g in train_graphs])
    node_scaler = StandardScaler().fit(all_train_node_features)
    for g in train_graphs:
        g.x = torch.tensor(node_scaler.transform(g.x.numpy()), dtype=torch.float)
    for g in test_graphs:
        g.x = torch.tensor(node_scaler.transform(g.x.numpy()), dtype=torch.float)

    for i, g in enumerate(train_graphs):
        g.y = torch.tensor(encoded_labels[i], dtype=torch.long)

    class_counts = np.bincount(encoded_labels, minlength=num_classes).astype(float)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=DEVICE)

    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)
    all_probabilities = []
    NUM_MODELS = int(os.environ.get("TC_CLASS_NUM_MODELS", "3"))

    for model_idx in range(NUM_MODELS):
        print(f"[tc-class] training ensemble model {model_idx + 1}/{NUM_MODELS}", flush=True)
        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)

        model = GNNClassifier(
            num_node_features=NUM_ATOM_FEATURES,
            num_edge_features=NUM_EXPANDED_EDGE_FEATURES,
            hidden_dim=HIDDEN_DIM,
            n_classes=num_classes,
        ).to(DEVICE)

        optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            epoch_batches = 0
            for batch in train_loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                logits = model(batch)
                loss = criterion(logits, batch.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                epoch_batches += 1
            scheduler.step()
            if LOG_EVERY > 0 and (
                epoch == 0 or (epoch + 1) % LOG_EVERY == 0 or epoch + 1 == EPOCHS
            ):
                avg_loss = epoch_loss / max(epoch_batches, 1)
                print(
                    f"[tc-class] model {model_idx + 1}/{NUM_MODELS} "
                    f"epoch {epoch + 1}/{EPOCHS} train_ce={avg_loss:.6f}",
                    flush=True,
                )

        model.eval()
        print(f"[tc-class] predicting with ensemble model {model_idx + 1}/{NUM_MODELS}", flush=True)
        model_probabilities = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                model_probabilities.extend(probs.cpu().numpy())
        all_probabilities.append(model_probabilities)

    avg_probabilities = np.mean(np.array(all_probabilities), axis=0)
    pred_indices = np.argmax(avg_probabilities, axis=1)
    predictions = label_encoder.inverse_transform(pred_indices).tolist()

    print("[tc-class] prediction complete", flush=True)
    return predictions
# EVOLVE-BLOCK-END


def run_classification(
    train_csv: str | Path,
    test_csv: str | Path,
    data_root: str | Path,
) -> dict[str, list]:
    train_rows = load_classification_rows(train_csv, data_root)
    test_rows = load_classification_rows(test_csv, data_root)
    predictions = fit_predict(train_rows, test_rows)

    return {
        "y_true": [row["label"] for row in test_rows],
        "y_pred": predictions,
        "test_rows": test_rows,
    }
