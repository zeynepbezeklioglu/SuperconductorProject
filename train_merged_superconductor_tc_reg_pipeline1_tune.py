import argparse
import os
import random
import warnings
from typing import Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress common warnings from pymatgen and other libraries
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=FutureWarning)

# --- PyTorch Imports ---
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print("PyTorch not found. The GNN model cannot be used.")
    torch = None


# --- Constants and Configuration ---
# Set device to CUDA if available, otherwise CPU
DEVICE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch else "cpu"
)

# Graph construction parameters
RADIUS_CUTOFF = 5.0  # Angstroms for neighbor search
MAX_ATOMIC_NUM = 100  # For feature table lookup

# --- Atom Feature Engineering ---
# Pre-computed, normalized features for atoms Z=1 to Z=100.
# Features: [group, row, atomic_mass, atomic_radius, X (electronegativity), ionization_energy]
NUM_NODE_FEATURES = 6
ATOM_FEATURES = {
    1: [-2.02, -1.53, -0.8, -1.5, -0.07, 0.44],
    2: [2.09, -1.53, -0.78, -1.61, 0.0, 2.37],
    3: [-2.02, -1.08, -0.76, 0.28, -1.45, -1.13],
    4: [-1.8, -1.08, -0.75, -0.21, -0.5, -0.45],
    5: [0.99, -1.08, -0.74, -0.7, 0.53, -0.63],
    6: [1.21, -1.08, -0.74, -0.92, 0.82, -0.11],
    7: [1.43, -1.08, -0.73, -1.07, 1.4, 0.61],
    8: [1.65, -1.08, -0.72, -1.18, 1.9, 0.44],
    9: [1.87, -1.08, -0.71, -1.29, 2.47, 1.12],
    10: [2.09, -1.08, -0.7, -1.33, 0.0, 1.83],
    11: [-2.02, -0.62, -0.68, 0.88, -1.53, -1.18],
    12: [-1.8, -0.62, -0.68, 0.43, -0.9, -0.74],
    13: [0.99, -0.62, -0.67, 0.17, -0.41, -1.02],
    14: [1.21, -0.62, -0.66, -0.04, -0.03, -0.66],
    15: [1.43, -0.62, -0.65, -0.28, 0.5, -0.25],
    16: [1.65, -0.62, -0.64, -0.4, 0.77, -0.27],
    17: [1.87, -0.62, -0.62, -0.51, 1.43, 0.33],
    18: [2.09, -0.62, -0.6, -0.55, 0.0, 0.83],
    19: [-2.02, -0.17, -0.6, 1.5, -1.71, -1.32],
    20: [-1.8, -0.17, -0.6, 1.09, -1.39, -1.0],
    21: [-1.58, -0.17, -0.57, 0.73, -0.85, -0.91],
    22: [-1.36, -0.17, -0.55, 0.52, -0.56, -0.87],
    23: [-1.14, -0.17, -0.53, 0.43, -0.38, -0.88],
    24: [-0.92, -0.17, -0.53, 0.32, -0.34, -0.87],
    25: [-0.7, -0.17, -0.51, 0.28, -0.52, -0.77],
    26: [-0.48, -0.17, -0.5, 0.25, -0.13, -0.69],
    27: [-0.26, -0.17, -0.48, 0.21, -0.04, -0.7],
    28: [-0.04, -0.17, -0.48, 0.17, 0.03, -0.73],
    29: [0.18, -0.17, -0.46, 0.28, -0.01, -0.72],
    30: [0.4, -0.17, -0.46, 0.23, -0.36, -0.43],
    31: [0.99, -0.17, -0.43, 0.17, -0.16, -1.02],
    32: [1.21, -0.17, -0.41, -0.01, 0.2, -0.69],
    33: [1.43, -0.17, -0.39, -0.19, 0.48, -0.34],
    34: [1.65, -0.17, -0.37, -0.3, 0.74, -0.35],
    35: [1.87, -0.17, -0.35, -0.42, 1.18, -0.03],
    36: [2.09, -0.17, -0.33, -0.44, 0.0, 0.52],
    37: [-2.02, 0.29, -0.31, 1.83, -1.73, -1.35],
    38: [-1.8, 0.29, -0.29, 1.34, -1.49, -1.05],
    39: [-1.58, 0.29, -0.28, 1.04, -1.01, -0.96],
    40: [-1.36, 0.29, -0.27, 0.81, -0.83, -0.9],
    41: [-1.14, 0.29, -0.25, 0.65, -0.43, -0.88],
    42: [-0.92, 0.29, -0.22, 0.54, 0.45, -0.82],
    43: [-0.7, 0.29, -0.2, 0.45, -0.01, -0.8],
    44: [-0.48, 0.29, -0.18, 0.4, 0.53, -0.78],
    45: [-0.26, 0.29, -0.16, 0.36, 0.56, -0.77],
    46: [-0.04, 0.29, -0.14, 0.32, 0.5, -0.62],
    47: [0.18, 0.29, -0.13, 0.43, 0.03, -0.74],
    48: [0.4, 0.29, -0.1, 0.4, -0.3, -0.49],
    49: [0.99, 0.29, -0.08, 0.36, -0.18, -1.04],
    50: [1.21, 0.29, -0.06, 0.17, 0.08, -0.78],
    51: [1.43, 0.29, -0.04, 0.0, 0.29, -0.58],
    52: [1.65, 0.29, -0.01, -0.13, 0.4, -0.48],
    53: [1.87, 0.29, 0.01, -0.25, 0.92, -0.22],
    54: [2.09, 0.29, 0.04, -0.26, 0.0, 0.17],
    55: [-2.02, 0.74, 0.06, 2.09, -1.8, -1.4],
    56: [-1.8, 0.74, 0.08, 1.63, -1.58, -1.16],
    57: [0.68, 0.74, 0.1, 1.15, -1.18, -1.07],
    58: [0.75, 0.74, 0.12, 1.15, -1.16, -1.06],
    59: [0.82, 0.74, 0.13, 1.14, -1.15, -1.05],
    60: [0.89, 0.74, 0.13, 1.13, -1.12, -1.04],
    61: [0.96, 0.74, 0.14, 1.12, -1.1, -1.04],
    62: [1.03, 0.74, 0.18, 1.11, -1.07, -1.05],
    63: [1.1, 0.74, 0.19, 1.1, -1.01, -1.04],
    64: [1.17, 0.74, 0.22, 1.09, -1.03, -1.0],
    65: [1.24, 0.74, 0.23, 1.07, -1.18, -0.99],
    66: [1.31, 0.74, 0.25, 1.06, -1.01, -0.99],
    67: [1.38, 0.74, 0.26, 1.05, -1.0, -0.98],
    68: [1.45, 0.74, 0.27, 1.04, -0.99, -0.98],
    69: [1.52, 0.74, 0.28, 1.03, -0.98, -0.97],
    70: [1.59, 0.74, 0.3, 1.02, -0.97, -0.44],
    71: [-1.58, 0.74, 0.31, 0.94, -0.96, -1.09],
    72: [-1.36, 0.74, 0.34, 0.81, -0.88, -0.89],
    73: [-1.14, 0.74, 0.36, 0.69, -0.59, -0.7],
    74: [-0.92, 0.74, 0.37, 0.58, 0.61, -0.68],
    75: [-0.7, 0.74, 0.4, 0.5, -0.01, -0.7],
    76: [-0.48, 0.74, 0.43, 0.45, 0.5, -0.56],
    77: [-0.26, 0.74, 0.43, 0.4, 0.52, -0.5],
    78: [-0.04, 0.74, 0.45, 0.36, 0.56, -0.48],
    79: [0.18, 0.74, 0.46, 0.4, 0.77, -0.45],
    80: [0.4, 0.74, 0.49, 0.4, -0.01, -0.25],
    81: [0.99, 0.74, 0.52, 0.36, -0.15, -1.0],
    82: [1.21, 0.74, 0.53, 0.21, 0.6, -0.77],
    83: [1.43, 0.74, 0.54, 0.1, 0.23, -0.78],
    84: [1.65, 0.74, 0.54, -0.05, 0.17, -0.61],
    85: [1.87, 0.74, 0.56, -0.17, 0.48, -0.37],
    86: [2.09, 0.74, 0.56, -0.16, 0.0, -0.14],
    87: [-2.02, 1.2, 0.56, 2.15, -1.58, -1.14],
    88: [-1.8, 1.2, 0.58, 1.83, -1.55, -1.18],
    89: [0.68, 1.2, 0.58, 1.27, -1.18, -1.1],
    90: [-1.36, 1.2, 0.62, 1.09, -0.88, -0.99],
    91: [-1.14, 1.2, 0.6, 0.94, -0.77, -0.91],
    92: [-0.92, 1.2, 0.65, 0.88, -0.77, -1.01],
    93: [-0.7, 1.2, 0.65, 0.81, -0.73, -0.9],
    94: [-0.48, 1.2, 0.68, 0.77, -0.92, -1.02],
    95: [-0.26, 1.2, 0.67, 0.73, -0.88, -1.02],
    96: [-0.04, 1.2, 0.69, 0.69, -0.87, -0.99],
    97: [0.18, 1.2, 0.69, 0.65, -0.85, -0.98],
    98: [0.4, 1.2, 0.72, 0.62, -0.83, -0.97],
    99: [0.99, 1.2, 0.73, 0.58, -0.82, -0.96],
    100: [1.21, 1.2, 0.73, 0.54, -0.8, -0.95],
}

# Model hyperparameters
EMBED_DIM = 64
HIDDEN_DIM = 128
N_GNN_LAYERS = 3
N_OUTPUTS = 1
NUM_EDGE_FEATURES = 50

# Training hyperparameters
N_EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5


# --- Graph Featurization ---
def _dummy_graph() -> Any:
    """
    Returns a minimal fallback graph so failed featurizations are not excluded.
    """
    node_features = torch.zeros((1, NUM_NODE_FEATURES), dtype=torch.float)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 1), dtype=torch.float)
    return (node_features, edge_index, edge_attr)


def featurize_cif(path: str) -> Any | None:
    """
    Parses a CIF file and constructs a graph representation with rich atom features.
    """
    try:
        from pymatgen.core.structure import Structure

        struct = Structure.from_file(path)

        node_features_list = []
        default_features = [0.0] * NUM_NODE_FEATURES
        for site in struct:
            if site.is_ordered:
                atomic_num = min(site.specie.Z, MAX_ATOMIC_NUM)
            else:
                main_element = max(site.species.items(), key=lambda x: x[1])[0]
                atomic_num = min(main_element.Z, MAX_ATOMIC_NUM)
            features = ATOM_FEATURES.get(atomic_num, default_features)
            node_features_list.append(features)

        if not node_features_list:
            return _dummy_graph()

        node_features = torch.tensor(node_features_list, dtype=torch.float)

        all_neighbors = struct.get_all_neighbors(r=RADIUS_CUTOFF, include_index=True)
        edge_src, edge_dst, edge_attr = [], [], []
        for i, neighbors in enumerate(all_neighbors):
            for _, dist, j, _ in neighbors:
                edge_src.append(i)
                edge_dst.append(j)
                edge_attr.append(dist)

        if not edge_src:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

        return (node_features, edge_index, edge_attr)

    except Exception:
        return _dummy_graph()


# --- PyTorch GNN Implementation ---
if torch:

    class GaussianExpansion(nn.Module):
        def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
            super().__init__()
            offsets = torch.linspace(start, stop, num_gaussians)
            self.gamma = -0.5 / (offsets[1] - offsets[0]).item() ** 2
            self.register_buffer("offsets", offsets)

        def forward(self, distances):
            return torch.exp(self.gamma * (distances - self.offsets) ** 2)


    class MessagePassingLayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, edge_dim: int):
            super().__init__()
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * in_dim + edge_dim, out_dim),
                nn.ReLU(),
            )
            self.node_update_mlp = nn.Sequential(
                nn.Linear(in_dim + out_dim, out_dim),
                nn.ReLU(),
            )
            self.norm = nn.LayerNorm(out_dim)

        def forward(self, x, edge_index, edge_attr):
            row, col = edge_index
            message_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
            messages = self.edge_mlp(message_input)

            aggregated_messages = torch.zeros(
                (x.size(0), messages.size(1)), device=x.device
            )
            aggregated_messages.scatter_add_(
                0, col.unsqueeze(-1).expand_as(messages), messages
            )

            update_input = torch.cat([x, aggregated_messages], dim=-1)
            x_new = self.node_update_mlp(update_input)

            return self.norm(x + x_new)


    class CrystalGNN(nn.Module):
        def __init__(
            self,
            embed_dim: int,
            hidden_dim: int,
            n_layers: int,
            n_outputs: int,
            edge_dim: int,
        ):
            super().__init__()
            self.feature_projection = nn.Linear(NUM_NODE_FEATURES, embed_dim)
            self.gaussian_expansion = GaussianExpansion(
                stop=RADIUS_CUTOFF, num_gaussians=edge_dim
            )
            self.layers = nn.ModuleList(
                [
                    MessagePassingLayer(embed_dim, embed_dim, edge_dim)
                    for _ in range(n_layers)
                ]
            )
            self.output_mlp = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_outputs),
            )

        def forward(self, node_features, edge_index, edge_attr, batch_idx):
            x = self.feature_projection(node_features)
            expanded_edge_attr = self.gaussian_expansion(edge_attr)

            for layer in self.layers:
                x = layer(x, edge_index, expanded_edge_attr)

            num_graphs = batch_idx.max().item() + 1
            graph_embedding = torch.zeros((num_graphs, x.size(1)), device=x.device)
            graph_embedding.scatter_add_(0, batch_idx.unsqueeze(-1).expand_as(x), x)
            node_counts = torch.bincount(batch_idx, minlength=num_graphs).unsqueeze(-1)
            graph_embedding = graph_embedding / node_counts.clamp(min=1)

            return self.output_mlp(graph_embedding)


    class CrystalGraphDataset(Dataset):
        def __init__(self, X: List, y: np.ndarray):
            self.X = X
            self.y = torch.tensor(y, dtype=torch.float)

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, idx: int) -> Tuple:
            return self.X[idx], self.y[idx]


    def collate_fn(batch: List[Tuple]) -> Tuple:
        graphs, targets = zip(*batch)
        node_features, edge_indices, edge_attrs = zip(*graphs)

        batch_idx = torch.cat(
            [
                torch.full((n.size(0),), i, dtype=torch.long)
                for i, n in enumerate(node_features)
            ]
        )
        cat_node_features = torch.cat(node_features, dim=0)
        cat_edge_attrs = torch.cat(edge_attrs, dim=0)

        node_counts = [n.size(0) for n in node_features]
        node_offsets = torch.tensor(
            [0] + list(np.cumsum(node_counts[:-1])), dtype=torch.long
        )

        cat_edge_indices_list = [
            edge_index + node_offsets[i] for i, edge_index in enumerate(edge_indices)
        ]
        cat_edge_indices = torch.cat(cat_edge_indices_list, dim=1)

        targets = torch.stack(targets)
        return cat_node_features, cat_edge_indices, cat_edge_attrs, batch_idx, targets


# --- Required Interface Functions ---
def build_features(df: pd.DataFrame) -> Tuple[List, np.ndarray]:
    X, y = [], []
    total_rows = len(df)
    progress_interval = max(1, total_rows // 20) if total_rows else 1

    print(f"Featurizing {total_rows} CIF files...")
    for processed_count, (idx, row) in enumerate(df.iterrows(), start=1):
        if (
            processed_count == 1
            or processed_count == total_rows
            or processed_count % progress_interval == 0
        ):
            print(f"  featurized {processed_count}/{total_rows}")

        graph_data = featurize_cif(row["cif"])
        if graph_data:
            X.append(graph_data)
            y.append(df.loc[idx, "Tc"])

    print(f"Successfully featurized {len(X)} of {total_rows} rows")
    return X, np.array(y)


def train_model(X: List, y: np.ndarray, model: Any = None, epochs: int = N_EPOCHS) -> Any:
    if not torch:
        raise RuntimeError("PyTorch is not available, cannot train the model.")

    dataset = CrystalGraphDataset(X, y)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    if model is None:
        model = CrystalGNN(
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            n_layers=N_GNN_LAYERS,
            n_outputs=N_OUTPUTS,
            edge_dim=NUM_EDGE_FEATURES,
        ).to(DEVICE)
    else:
        model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.L1Loss()

    print(f"Training GNN regressor for {epochs} epochs on device: {DEVICE}")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            batch_tensors = {
                k: v.to(DEVICE)
                for k, v in zip(
                    ["nodes", "edges", "edge_attr", "batch_idx", "targets"], batch
                )
                if isinstance(v, torch.Tensor)
            }

            node_features = batch_tensors["nodes"]
            edge_index = batch_tensors["edges"]
            edge_attr = batch_tensors["edge_attr"]
            batch_idx = batch_tensors["batch_idx"]
            targets = batch_tensors["targets"]

            optimizer.zero_grad()
            predictions = model(node_features, edge_index, edge_attr, batch_idx)
            loss = criterion(predictions.squeeze(), targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if (
            epoch == 0
            or epoch == epochs - 1
            or (epoch + 1) % max(1, epochs // 10) == 0
        ):
            mean_loss = epoch_loss / num_batches if num_batches else float("nan")
            print(f"  epoch {epoch + 1}/{epochs} - mean loss: {mean_loss:.6f}")

    print("Model training complete")
    return model


def predict(model: Any, X: List) -> np.ndarray:
    """Runs inference with the trained GNN regressor and returns predicted Tc.

    Uses the exact same batch handling as train_model (no change to GNN logic);
    only difference is eval mode + no gradients + continuous output.
    """
    if not torch:
        raise RuntimeError("PyTorch is not available, cannot run predictions.")
    if model is None or not X:
        return np.array([])

    model = model.to(DEVICE)
    model.eval()

    dummy_y = np.zeros(len(X), dtype=float)
    dataset = CrystalGraphDataset(X, dummy_y)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    all_preds: List[float] = []
    total_batches = len(dataloader)
    print(f"Running prediction across {total_batches} batches...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, start=1):
            batch_tensors = {
                k: v.to(DEVICE)
                for k, v in zip(
                    ["nodes", "edges", "edge_attr", "batch_idx", "targets"], batch
                )
                if isinstance(v, torch.Tensor)
            }
            node_features = batch_tensors["nodes"]
            edge_index = batch_tensors["edges"]
            edge_attr = batch_tensors["edge_attr"]
            batch_idx_t = batch_tensors["batch_idx"]

            outputs = model(node_features, edge_index, edge_attr, batch_idx_t)
            all_preds.extend(
                np.atleast_1d(outputs.squeeze().cpu().numpy()).reshape(-1).tolist()
            )

            if (
                batch_idx == 1
                or batch_idx == total_batches
                or batch_idx % max(1, total_batches // 10) == 0
            ):
                print(f"  predicted batch {batch_idx}/{total_batches}")

    return np.array(all_preds)


def resolve_local_cif_path(
    original_path: str,
    extracted_dir: str,
    split_dir: str,
    train_test_label: str | None = None,
) -> str:
    """
    Maps the CIF path stored in the CSV to a local CIF file.
    """
    if not isinstance(original_path, str) or not original_path:
        return original_path

    filename = os.path.basename(original_path)

    extracted_candidate = os.path.join(extracted_dir, filename)
    if os.path.exists(extracted_candidate):
        return extracted_candidate

    if train_test_label:
        subset = str(train_test_label).strip().lower()
        labeled_candidate = os.path.join(split_dir, subset, filename)
        if os.path.exists(labeled_candidate):
            return labeled_candidate

    for subset in ("train", "test", "unlabeled"):
        split_candidate = os.path.join(split_dir, subset, filename)
        if os.path.exists(split_candidate):
            return split_candidate

    return original_path


def load_dataset(csv_path: str, extracted_dir: str, split_dir: str) -> pd.DataFrame:
    """
    Loads merged_superconductor_data.csv and resolves local CIF paths.
    """
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path).copy()
    print(f"Loaded {len(df)} rows")

    required_columns = {"cif", "Tc"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    print("Resolving CIF paths to local files...")
    total_rows = len(df)
    progress_interval = max(1, total_rows // 20) if total_rows else 1
    resolved_paths = []
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        if idx == 1 or idx == total_rows or idx % progress_interval == 0:
            print(f"  resolved {idx}/{total_rows} CIF paths")
        resolved_paths.append(
            resolve_local_cif_path(
                original_path=row["cif"],
                extracted_dir=extracted_dir,
                split_dir=split_dir,
                train_test_label=row.get("train_test_label"),
            )
        )

    df["cif"] = resolved_paths
    missing_cifs = int((~df["cif"].apply(os.path.exists)).sum())
    print(f"Finished CIF resolution. Missing local CIF files: {missing_cifs}")
    return df


def drop_invalid_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops rows with non-finite Tc values and reports how many were removed.
    """
    df = df.copy()
    original_count = len(df)
    df["Tc"] = pd.to_numeric(df["Tc"], errors="coerce")
    valid_mask = np.isfinite(df["Tc"])
    removed_count = int((~valid_mask).sum())
    if removed_count:
        print(f"Dropping {removed_count} rows with invalid Tc values")
    filtered_df = df.loc[valid_mask].copy()
    print(f"Rows after Tc filtering: {len(filtered_df)} / {original_count}")
    return filtered_df


def set_seed(seed: int) -> None:
    """Fix all RNGs for reproducible runs (does not change model logic)."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def main():
    global EMBED_DIM, HIDDEN_DIM, N_GNN_LAYERS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE
    parser = argparse.ArgumentParser(
        description=(
            "Train the TC_reg_pipeline_1 GNN regressor with a train/test split, "
            "evaluate on the held-out set (R2/MAE/RMSE), and export predictions."
        )
    )
    parser.add_argument(
        "--csv",
        default="merged_superconductor_data.csv",
        help="Path to merged_superconductor_data.csv",
    )
    parser.add_argument(
        "--extracted-cif-dir",
        default="extracted_merged_superconductor_cifs",
        help="Directory containing extracted CIF files",
    )
    parser.add_argument(
        "--split-cif-dir",
        default="split_cifs_by_csv_label",
        help="Directory containing train/test/unlabeled CIF subdirectories",
    )
    parser.add_argument(
        "--output-model",
        default="tc_reg_pipeline1_gnn_model.joblib",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--resume-model",
        default=None,
        help="Optional existing model checkpoint to resume training from",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start a fresh training run even if the output model already exists",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=N_EPOCHS,
        help="Number of epochs to train for in this run",
    )
    parser.add_argument(
        "--save-test-predictions",
        default="tc_reg_test_predictions.csv",
        help="CSV path for held-out test predictions",
    )
    parser.add_argument(
        "--year-cutoff",
        type=int,
        default=None,
        help=(
            "Optional temporal split: rows with year < cutoff become train, "
            "year >= cutoff become test. Requires a 'year' column in the CSV. "
            "If omitted, the split comes from the 'train_test_label' column."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (weights init + batch shuffling)",
    )
    parser.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--layers", type=int, default=N_GNN_LAYERS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # Override module-level hyperparameters from CLI (train_model reads these globals)
    EMBED_DIM = args.embed_dim
    HIDDEN_DIM = args.hidden_dim
    N_GNN_LAYERS = args.layers
    LEARNING_RATE = args.lr
    WEIGHT_DECAY = args.weight_decay
    BATCH_SIZE = args.batch_size
    print(
        f"Hyperparameters: embed_dim={EMBED_DIM}, hidden_dim={HIDDEN_DIM}, "
        f"layers={N_GNN_LAYERS}, lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}, "
        f"batch_size={BATCH_SIZE}, epochs={args.epochs}"
    )

    set_seed(args.seed)

    csv_path = os.path.abspath(args.csv)
    extracted_cif_dir = os.path.abspath(args.extracted_cif_dir)
    split_cif_dir = os.path.abspath(args.split_cif_dir)
    output_model = os.path.abspath(args.output_model)
    save_test_predictions = os.path.abspath(args.save_test_predictions)
    resume_model = os.path.abspath(args.resume_model) if args.resume_model else None
    if resume_model is None and not args.fresh and os.path.exists(output_model):
        resume_model = output_model

    print("Starting TC_reg_pipeline_1 train/eval run")
    print(f"CSV path: {csv_path}")
    print(f"Extracted CIF directory: {extracted_cif_dir}")
    print(f"Split CIF directory: {split_cif_dir}")
    print(f"Output model path: {output_model}")
    print(f"Epochs this run: {args.epochs}")
    if resume_model:
        print(f"Resume model path: {resume_model}")
    elif args.fresh:
        print("Fresh training requested; existing checkpoint will be ignored")

    df = load_dataset(
        csv_path=csv_path,
        extracted_dir=extracted_cif_dir,
        split_dir=split_cif_dir,
    )
    df = drop_invalid_targets(df)
    if df.empty:
        raise ValueError("No valid rows found in the dataset.")

    # --- Train/test split (random via train_test_label, or temporal via --year-cutoff) ---
    if args.year_cutoff is not None:
        if "year" not in df.columns:
            raise ValueError(
                "--year-cutoff was given but the CSV has no 'year' column."
            )
        print(f"Using TEMPORAL split at year cutoff {args.year_cutoff}")
        years = pd.to_numeric(df["year"], errors="coerce")
        train_df = df[years < args.year_cutoff].copy()
        test_df = df[years >= args.year_cutoff].copy()
    elif "train_test_label" in df.columns:
        print("Using RANDOM split from 'train_test_label' column")
        train_df = df[df["train_test_label"].astype(str).str.lower() == "train"].copy()
        test_df = df[df["train_test_label"].astype(str).str.lower() == "test"].copy()
    else:
        print("No split information found; using all rows for training")
        train_df = df.copy()
        test_df = pd.DataFrame(columns=df.columns)

    if train_df.empty:
        raise ValueError("No training rows found for the chosen split.")

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    print("Building training features...")
    X_train, y_train = build_features(train_df)
    if not X_train:
        raise ValueError("No training features could be extracted from the CIF files.")

    existing_model = None
    if resume_model:
        print(f"Loading existing model from: {resume_model}")
        existing_model = joblib.load(resume_model)

    print("Training model...")
    model = train_model(X_train, y_train, model=existing_model, epochs=args.epochs)

    os.makedirs(os.path.dirname(output_model) or ".", exist_ok=True)
    print(f"Saving model to: {output_model}")
    joblib.dump(model, output_model)
    print(f"Saved model to: {output_model}")

    # --- Evaluation on the held-out test set ---
    if not test_df.empty:
        print("Building test features...")
        X_test, y_test = build_features(test_df)
        if X_test:
            print("Generating test predictions...")
            y_pred = predict(model, X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = r2_score(y_test, y_pred)
            print("==================== TEST METRICS ====================")
            print(f"Test rows used: {len(y_test)}")
            print(f"R2:   {r2:.4f}")
            print(f"MAE:  {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print("======================================================")

            test_results = pd.DataFrame(
                {
                    "true_Tc": np.asarray(y_test).reshape(-1),
                    "predicted_Tc": np.asarray(y_pred).reshape(-1),
                }
            )
            test_results["abs_error"] = (
                test_results["true_Tc"] - test_results["predicted_Tc"]
            ).abs()
            os.makedirs(os.path.dirname(save_test_predictions) or ".", exist_ok=True)
            print(f"Saving test predictions to: {save_test_predictions}")
            test_results.to_csv(save_test_predictions, index=False)
            print("Test predictions saved")
        else:
            print("No test features could be extracted; skipping evaluation.")
    else:
        print("Test set is empty; no evaluation performed.")

    print("TC_reg_pipeline_1 train/eval run is complete")


if __name__ == "__main__":
    main()
