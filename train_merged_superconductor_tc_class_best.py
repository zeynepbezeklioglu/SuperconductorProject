import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pymatgen.core.structure import Structure
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Suppress pymatgen warnings and any other potential user warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Constants and Hyperparameters ---
CUTOFF_RADIUS = 5.0
HIDDEN_DIM = 64
NUM_GNN_LAYERS = 3
DROPOUT_RATE = 0.5
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- GNN Model Implementation ---
class GCNLayer(nn.Module):
    """A simple graph convolutional layer that incorporates edge weights."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.root_embed = nn.Linear(in_dim, out_dim)

    def forward(self, node_feats, edge_index, edge_weights):
        source_nodes, dest_nodes = edge_index
        neighbor_feats = node_feats[source_nodes]

        # Weight the neighbor features by edge weights
        # unsqueeze to broadcast weights across the feature dimension
        weighted_neighbor_feats = neighbor_feats * edge_weights.unsqueeze(1)

        # Sum aggregation of weighted neighbor features
        aggregated_feats = torch.zeros_like(node_feats)
        aggregated_feats.index_add_(0, dest_nodes, weighted_neighbor_feats)

        # Update node features
        out = self.root_embed(node_feats) + self.linear(aggregated_feats)
        return out


class CrystalGNN(nn.Module):
    """
    Crystal Graph Neural Network for class prediction.
    This model supports uncertainty estimation via Monte Carlo dropout.
    """

    def __init__(
        self,
        num_classes,
        atom_feature_dim,
        hidden_dim=HIDDEN_DIM,
        n_layers=NUM_GNN_LAYERS,
        dropout_rate=DROPOUT_RATE,
    ):
        super().__init__()
        self.atom_embedding = nn.Linear(atom_feature_dim, hidden_dim)

        self.conv_layers = nn.ModuleList(
            [GCNLayer(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)]
        )

        # Update MLP head to handle concatenated mean and max pooling
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        # Extract edge_weights from the data dictionary
        node_feats, edge_index, batch_ptr, edge_weights = (
            data["node_feats"],
            data["edge_index"],
            data["batch_ptr"],
            data["edge_weights"],
        )

        x = self.atom_embedding(node_feats)

        for conv, bn in zip(self.conv_layers, self.bn_layers):
            identity = x
            # Pass edge_weights to the convolutional layer
            x = conv(x, edge_index, edge_weights)
            x = bn(x)
            x = torch.relu(x)
            x = x + identity

        # Dual (mean + max) pooling
        graph_embeds = torch.zeros(len(batch_ptr) - 1, x.size(1) * 2, device=x.device)
        for i in range(len(batch_ptr) - 1):
            start, end = batch_ptr[i], batch_ptr[i + 1]
            if end > start:
                nodes = x[start:end]
                mean_pooled = nodes.mean(dim=0)
                max_pooled = nodes.max(dim=0).values
                graph_embeds[i] = torch.cat([mean_pooled, max_pooled])

        out = self.readout_mlp(graph_embeds)
        return out

    def enable_dropout(self):
        """Enables dropout layers for Monte Carlo uncertainty estimation during inference."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


# --- Data Handling ---
class CrystalGraphDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collate_fn(batch):
    """Collates a list of graph data into a single batch."""
    node_feats_list, edge_index_list, edge_weights_list, labels = [], [], [], []
    node_offset = 0
    batch_ptr_list = [0]

    for graph, label in batch:
        if graph is None:
            continue

        num_nodes = graph["node_feats"].shape[0]
        node_feats_list.append(graph["node_feats"])
        edge_index_list.append(graph["edge_index"] + node_offset)
        edge_weights_list.append(graph["edge_weights"])
        labels.append(label)
        node_offset += num_nodes
        batch_ptr_list.append(node_offset)

    if not node_feats_list:
        return None, None

    return {
        "node_feats": torch.cat(node_feats_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_weights": torch.cat(edge_weights_list, dim=0),
        "batch_ptr": torch.tensor(batch_ptr_list, dtype=torch.long),
    }, torch.tensor(labels, dtype=torch.long)


# --- Required Interface Functions ---
def featurize_cif(path: str):
    """
    Parses a CIF file and builds a graph representation with rich atomic features
    and distance-based edge weights.
    """
    try:
        struct = Structure.from_file(path)
        if not struct.is_ordered:
            return None

        # Build node features from elemental properties
        atom_features = []
        for site in struct:
            element = site.specie
            # Use a vector of atomic properties
            feature = [
                element.Z,
                element.group,
                element.row,
                float(element.atomic_mass),
                float(element.atomic_radius) if element.atomic_radius else 0.0,
                element.X if element.X else 0.0,
            ]
            atom_features.append(feature)
        node_feats = torch.tensor(atom_features, dtype=torch.float)

        # Build edge index and edge weights from crystal graph
        all_neighbors = struct.get_all_neighbors(r=CUTOFF_RADIUS, include_index=True)
        edge_src, edge_dst, edge_weights = [], [], []
        for i, neighbors in enumerate(all_neighbors):
            for neighbor in neighbors:
                j = neighbor.index
                dist = neighbor.nn_distance
                edge_src.append(i)
                edge_dst.append(j)
                # Calculate weight as exponential decay of distance
                edge_weights.append(np.exp(-dist))

        if not edge_src:  # Handle case with no neighbors found
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weights_tensor = torch.empty((0,), dtype=torch.float)
        else:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            edge_weights_tensor = torch.tensor(edge_weights, dtype=torch.float)

        return {
            "node_feats": node_feats,
            "edge_index": edge_index,
            "edge_weights": edge_weights_tensor,
        }
    except Exception:
        return None


def build_features(df):
    """Processes the manifest DataFrame to create features (X) and labels (y)."""
    X_graphs = []
    y_labels = []
    total_rows = len(df)
    progress_interval = max(1, total_rows // 20) if total_rows else 1

    print(f"Building graph features for {total_rows} rows...")
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        if idx == 1 or idx == total_rows or idx % progress_interval == 0:
            print(f"  featurized {idx}/{total_rows}")
        graph = featurize_cif(row["cif"])
        if graph is not None:
            X_graphs.append(graph)
            y_labels.append(row["class"])

    print(f"Successfully built {len(X_graphs)} graphs out of {total_rows} rows")
    return X_graphs, pd.Series(y_labels, dtype="category")


def train_model(X, y):
    """Trains the GNN model."""
    if not X:
        print("No training graphs available; skipping model training")
        return None

    print("Encoding class labels...")
    class_map = {label: i for i, label in enumerate(y.cat.categories)}
    inv_class_map = {i: label for label, i in class_map.items()}
    y_encoded = y.map(class_map).values
    num_classes = len(class_map)
    print(f"Detected {num_classes} classes")

    # --- Feature Scaling ---
    # Ensure there's at least one graph to get features from
    valid_graphs = [g for g in X if g is not None and g["node_feats"].numel() > 0]
    if not valid_graphs:
        print("No valid graphs with node features available after featurization")
        return None

    print(f"Scaling node features using {len(valid_graphs)} valid graphs...")
    all_node_feats = np.vstack([g["node_feats"].numpy() for g in valid_graphs])
    scaler = StandardScaler()
    scaler.fit(all_node_feats)
    for g in X:
        if g is not None and g["node_feats"].numel() > 0:
            g["node_feats"] = torch.from_numpy(
                scaler.transform(g["node_feats"].numpy())
            ).float()

    atom_feature_dim = X[0]["node_feats"].shape[1]
    print(f"Atom feature dimension: {atom_feature_dim}")

    dataset = CrystalGraphDataset(X, y_encoded)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    print(f"Created training dataloader with batch size {BATCH_SIZE}")

    model = CrystalGNN(num_classes=num_classes, atom_feature_dim=atom_feature_dim).to(
        DEVICE
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting training for {EPOCHS} epochs on device: {DEVICE}")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        num_batches = 0
        for data, labels in loader:
            if data is None:
                continue

            data = {k: v.to(DEVICE) for k, v in data.items()}
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if (
            epoch == 0
            or epoch == EPOCHS - 1
            or (epoch + 1) % max(1, EPOCHS // 10) == 0
        ):
            mean_loss = epoch_loss / num_batches if num_batches else float("nan")
            print(f"  epoch {epoch + 1}/{EPOCHS} - mean loss: {mean_loss:.6f}")

    print("Training complete")
    return {
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "inv_class_map": inv_class_map,
        "atom_feature_dim": atom_feature_dim,
        "scaler": scaler,
    }


def predict(model_package, X):
    """Makes predictions using the trained GNN model."""
    if not model_package:
        print("No model package provided for prediction")
        return np.array([])
    if not X:
        print("No input graphs provided for prediction")
        return np.array([])

    print(f"Preparing {len(X)} graphs for prediction...")
    scaler = model_package["scaler"]
    for g in X:
        if g is not None and g["node_feats"].numel() > 0:
            g["node_feats"] = torch.from_numpy(
                scaler.transform(g["node_feats"].numpy())
            ).float()

    num_classes = model_package["num_classes"]
    inv_class_map = model_package["inv_class_map"]
    atom_feature_dim = model_package["atom_feature_dim"]

    model = CrystalGNN(num_classes=num_classes, atom_feature_dim=atom_feature_dim).to(
        DEVICE
    )
    model.load_state_dict(model_package["model_state_dict"])
    model.eval()

    dummy_y = [0] * len(X)
    dataset = CrystalGraphDataset(X, dummy_y)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    all_preds = []
    total_batches = len(loader)
    print(f"Running prediction across {total_batches} batches...")
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader, start=1):
            if data is None:
                continue

            data = {k: v.to(DEVICE) for k, v in data.items()}
            output = model(data)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            all_preds.extend(preds)

            if (
                batch_idx == 1
                or batch_idx == total_batches
                or batch_idx % max(1, total_batches // 10) == 0
            ):
                print(f"  predicted batch {batch_idx}/{total_batches}")

    return [inv_class_map[p] for p in all_preds]


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
    Loads merged_superconductor_data.csv and adapts it to the logged classifier interface.
    """
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path).copy()
    print(f"Loaded {len(df)} rows")

    required_columns = {"label", "cif"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    print("Renaming 'label' column to 'class' for classifier compatibility...")
    df = df.rename(columns={"label": "class"})

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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the logged Tc class GNN solution on merged_superconductor_data.csv."
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
        default="tc_class_best_gnn_model_package.pt",
        help="Path to save the trained model package",
    )
    parser.add_argument(
        "--save-test-predictions",
        default="tc_class_test_predictions.csv",
        help="Optional CSV path for test predictions",
    )
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    extracted_cif_dir = os.path.abspath(args.extracted_cif_dir)
    split_cif_dir = os.path.abspath(args.split_cif_dir)
    output_model = os.path.abspath(args.output_model)
    save_test_predictions = os.path.abspath(args.save_test_predictions)

    print("Starting Tc class training setup")
    print(f"CSV path: {csv_path}")
    print(f"Extracted CIF directory: {extracted_cif_dir}")
    print(f"Split CIF directory: {split_cif_dir}")
    print(f"Output model path: {output_model}")

    df = load_dataset(
        csv_path=csv_path,
        extracted_dir=extracted_cif_dir,
        split_dir=split_cif_dir,
    )

    if "train_test_label" in df.columns:
        print("Using train/test split from 'train_test_label' column")
        train_df = df[df["train_test_label"].astype(str).str.lower() == "train"].copy()
        test_df = df[df["train_test_label"].astype(str).str.lower() == "test"].copy()
    else:
        print("No 'train_test_label' column found; using all rows for training")
        train_df = df.copy()
        test_df = pd.DataFrame(columns=df.columns)

    if train_df.empty:
        raise ValueError("No training rows found in the dataset.")

    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    print("Building training graphs...")
    X_train, y_train = build_features(train_df)
    if not X_train:
        raise ValueError("No training graphs could be built from the CIF files.")

    print("Training model package...")
    model_package = train_model(X_train, y_train)
    if not model_package:
        raise ValueError("Model training did not produce a valid model package.")

    os.makedirs(os.path.dirname(output_model) or ".", exist_ok=True)
    print(f"Saving trained model package to: {output_model}")
    torch.save(model_package, output_model)
    print("Model package saved")

    if not test_df.empty:
        print("Building test graphs...")
        X_test, y_test = build_features(test_df)
        if X_test:
            print("Generating test predictions...")
            y_pred = predict(model_package, X_test)
            test_results = pd.DataFrame(
                {
                    "true_class": y_test.reset_index(drop=True),
                    "predicted_class": y_pred,
                }
            )
            os.makedirs(os.path.dirname(save_test_predictions) or ".", exist_ok=True)
            print(f"Saving test predictions to: {save_test_predictions}")
            test_results.to_csv(save_test_predictions, index=False)
            print("Test predictions saved")
        else:
            print("No test graphs could be built; skipping prediction export")

    print("Tc class training script setup is complete")


if __name__ == "__main__":
    main()
