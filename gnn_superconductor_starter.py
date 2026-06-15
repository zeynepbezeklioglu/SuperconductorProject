"""
Starter Graph Neural Network for the superconductor project.

Supports:
  1) Tc regression
  2) Superconductor classification

"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pymatgen.core import Structure

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


ATOM_FEATURES = ["Z", "X", "atomic_mass", "atomic_radius", "group", "row"]


def safe_float(x, default=0.0):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def atom_features(element) -> list[float]:
    return [
        safe_float(element.Z) / 100.0,
        safe_float(element.X) / 4.0,
        safe_float(element.atomic_mass) / 250.0,
        safe_float(element.atomic_radius) / 3.0,
        safe_float(element.group) / 18.0,
        safe_float(element.row) / 7.0,
    ]


def resolve_local_cif_path(original_path: str, split_dir: str, train_test_label: str | None = None) -> str:
    if not isinstance(original_path, str) or not original_path:
        return original_path

    filename = os.path.basename(original_path)

    if train_test_label:
        subset = str(train_test_label).strip().lower()
        candidate = os.path.join(split_dir, subset, filename)
        if os.path.exists(candidate):
            return candidate

    for subset in ("train", "test"):
        candidate = os.path.join(split_dir, subset, filename)
        if os.path.exists(candidate):
            return candidate

    return original_path


@dataclass
class CrystalGraph:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor
    formula: str
    year: float | int | str


def cif_to_graph(cif_path: str, target: float | int, formula: str = "", year: float | int | str = "", cutoff: float = 5.0) -> CrystalGraph | None:
    try:
        structure = Structure.from_file(cif_path)
        node_features = []
        for site in structure:
            node_features.append(atom_features(site.specie))

        edges_src = []
        edges_dst = []
        edge_dists = []

        for i, site in enumerate(structure):
            neighbors = structure.get_neighbors(site, cutoff)
            for n in neighbors:
                j = int(n.index)
                if i == j:
                    continue
                edges_src.append(i)
                edges_dst.append(j)
                edge_dists.append([float(n.nn_distance) / cutoff])

        if not edges_src:
            return None

        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_dists, dtype=torch.float32)
        y = torch.tensor([target], dtype=torch.float32)

        return CrystalGraph(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, formula=formula, year=year)
    except Exception:
        return None


class SuperconductorGraphDataset(Dataset):
    def __init__(self, csv_path: str, split_cif_dir: str, split: str, task: str, cutoff: float = 5.0):
        self.csv_path = os.path.abspath(csv_path)
        self.split_cif_dir = os.path.abspath(split_cif_dir)
        self.split = split
        self.task = task
        self.cutoff = cutoff

        df = pd.read_csv(self.csv_path).copy()
        if "train_test_label" not in df.columns:
            raise ValueError("CSV must contain train_test_label column.")

        df = df[df["train_test_label"].astype(str).str.lower() == split].copy()

        if task == "regression":
            df["Tc"] = pd.to_numeric(df["Tc"], errors="coerce")
            df = df[np.isfinite(df["Tc"])].copy()
            target_col = "Tc"
        elif task == "classification":
            df["label"] = pd.to_numeric(df["label"], errors="coerce")
            df = df[np.isfinite(df["label"])].copy()
            target_col = "label"
        else:
            raise ValueError("task must be regression or classification")

        df["resolved_cif"] = df.apply(
            lambda row: resolve_local_cif_path(row["cif"], self.split_cif_dir, row.get("train_test_label")), axis=1
        )
        df = df[df["resolved_cif"].apply(os.path.exists)].copy()
        self.df = df.reset_index(drop=True)
        self.target_col = target_col

        print(f"{split} {task}: {len(self.df)} rows after target and CIF filtering")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return cif_to_graph(
            row["resolved_cif"],
            row[self.target_col],
            formula=row.get("formula", ""),
            year=row.get("year", ""),
            cutoff=self.cutoff,
        )


def collate_graphs(batch):
    batch = [g for g in batch if g is not None]
    if not batch:
        return None

    xs, edge_indices, edge_attrs, ys, graph_ids = [], [], [], [], []
    node_offset = 0
    formulas, years = [], []

    for graph_id, g in enumerate(batch):
        n = g.x.shape[0]
        xs.append(g.x)
        edge_indices.append(g.edge_index + node_offset)
        edge_attrs.append(g.edge_attr)
        ys.append(g.y)
        graph_ids.append(torch.full((n,), graph_id, dtype=torch.long))
        formulas.append(g.formula)
        years.append(g.year)
        node_offset += n

    return {
        "x": torch.cat(xs, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "batch": torch.cat(graph_ids, dim=0),
        "y": torch.cat(ys, dim=0),
        "formula": formulas,
        "year": years,
    }


class SimpleCrystalGNN(nn.Module):
    def __init__(self, node_dim: int = 6, hidden_dim: int = 64, num_layers: int = 3, task: str = "regression"):
        super().__init__()
        self.task = task
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.update_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        out_dim = 1 if task == "regression" else 2
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_embed(x)
        src, dst = edge_index

        for edge_mlp, update_mlp in zip(self.edge_mlps, self.update_mlps):
            m_in = torch.cat([h[src], h[dst], edge_attr], dim=1)
            messages = edge_mlp(m_in)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, messages)
            h = h + update_mlp(torch.cat([h, agg], dim=1))

        num_graphs = int(batch.max().item()) + 1
        graph_emb = torch.zeros(num_graphs, h.shape[1], device=h.device)
        graph_emb.index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=num_graphs).float().unsqueeze(1).to(h.device)
        graph_emb = graph_emb / counts.clamp_min(1.0)
        return self.head(graph_emb)


def evaluate(model, loader, task, device):
    model.eval()
    y_true, y_pred = [], []
    rows = []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            graph_batch = batch["batch"].to(device)
            y = batch["y"].to(device)
            out = model(x, edge_index, edge_attr, graph_batch)

            if task == "regression":
                pred = out.squeeze(1)
                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())
            else:
                pred = out.argmax(dim=1)
                y_true.extend(y.long().cpu().numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())

            for formula, year, true_val, pred_val in zip(batch["formula"], batch["year"], y_true[-len(batch["formula"]):], y_pred[-len(batch["formula"]):]):
                rows.append({"formula": formula, "year": year, "true": true_val, "predicted": pred_val})

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if task == "regression":
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return {"MAE": mae, "RMSE": rmse, "R2": r2}, rows
    else:
        acc = float((y_true == y_pred).mean())
        return {"accuracy": acc}, rows


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SuperconductorGraphDataset(args.csv, args.split_cif_dir, "train", args.task, args.cutoff)
    test_ds = SuperconductorGraphDataset(args.csv, args.split_cif_dir, "test", args.task, args.cutoff)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)

    model = SimpleCrystalGNN(hidden_dim=args.hidden_dim, num_layers=args.layers, task=args.task).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            if batch is None:
                continue
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            graph_batch = batch["batch"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            out = model(x, edge_index, edge_attr, graph_batch)
            if args.task == "regression":
                loss = F.l1_loss(out.squeeze(1), y)
            else:
                loss = F.cross_entropy(out, y.long())
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics, _ = evaluate(model, test_loader, args.task, device)
            print(f"Epoch {epoch:03d} | train_loss={total_loss/max(n_batches,1):.4f} | test={metrics}")

    metrics, rows = evaluate(model, test_loader, args.task, device)
    print("Final test metrics:", metrics)

    out_csv = f"gnn_{args.task}_predictions.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    torch.save(model.state_dict(), f"gnn_{args.task}_model.pt")
    print(f"Saved predictions to {out_csv}")
    print(f"Saved model to gnn_{args.task}_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["regression", "classification"], required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--split-cif-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=5)
    args = parser.parse_args()
    train(args)
