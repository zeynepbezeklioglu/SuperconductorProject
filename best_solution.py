# feature_builder_candidate.py

import pandas as pd
import numpy as np
import warnings
from typing import List, Tuple, Any

# Suppress common warnings from pymatgen and other libraries
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=FutureWarning)

# --- PyTorch Imports ---
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
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

# --- New: Atom Feature Engineering ---
# Pre-computed, normalized features for atoms Z=1 to Z=100.
# Features: [group, row, atomic_mass, atomic_radius, X (electronegativity), ionization_energy]
# Data is z-score normalized based on properties of the first 100 elements.
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
N_EPOCHS = 120
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# --- Graph Featurization ---


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
            atomic_num = min(site.specie.Z, MAX_ATOMIC_NUM)
            features = ATOM_FEATURES.get(atomic_num, default_features)
            node_features_list.append(features)

        node_features = torch.tensor(node_features_list, dtype=torch.float)

        all_neighbors = struct.get_all_neighbors(r=RADIUS_CUTOFF, include_index=True)
        edge_src, edge_dst, edge_attr = [], [], []
        for i, neighbors in enumerate(all_neighbors):
            for _, dist, j, _ in neighbors:
                edge_src.append(i)
                edge_dst.append(j)
                edge_attr.append(dist)

        if not edge_src:
            return None

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

        return (node_features, edge_index, edge_attr)

    except Exception:
        return None


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
            # --- MODIFICATION: Revert to single embedding size for mean pooling only ---
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

            # --- MODIFICATION: Use only mean pooling ---
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
    # Pre-featurize to avoid repeated calls in a loop
    featurized_cifs = df["cif"].apply(featurize_cif)

    for idx, graph_data in featurized_cifs.items():
        if graph_data:
            X.append(graph_data)
            y.append(df.loc[idx, "Tc"])

    return X, np.array(y)


def train_model(X: List, y: np.ndarray) -> Any:
    if not torch:
        raise RuntimeError("PyTorch is not available, cannot train the model.")

    dataset = CrystalGraphDataset(X, y)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    model = CrystalGNN(
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_GNN_LAYERS,
        n_outputs=N_OUTPUTS,
        edge_dim=NUM_EDGE_FEATURES,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.L1Loss()  # MAE loss

    model.train()
    for epoch in range(N_EPOCHS):
        for batch in dataloader:
            # Move only tensors to device
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

    return model


def predict(model: Any, X: List) -> np.ndarray:
    if not torch:
        raise RuntimeError("PyTorch is not available, cannot make predictions.")

    y_dummy = np.zeros(len(X))
    dataset = CrystalGraphDataset(X, y_dummy)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    model.to(DEVICE)
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            # Move only tensors to device, skip target
            batch_tensors = {
                k: v.to(DEVICE)
                for k, v in zip(
                    ["nodes", "edges", "edge_attr", "batch_idx"], batch[:-1]
                )
                if isinstance(v, torch.Tensor)
            }

            node_features = batch_tensors["nodes"]
            edge_index = batch_tensors["edges"]
            edge_attr = batch_tensors["edge_attr"]
            batch_idx = batch_tensors["batch_idx"]

            predictions = model(node_features, edge_index, edge_attr, batch_idx)
            all_predictions.append(predictions.cpu().numpy())

    return np.concatenate(all_predictions).squeeze()
