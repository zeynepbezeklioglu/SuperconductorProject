from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from pymatgen.core import Composition, Element, Structure
except Exception as exc:  # pragma: no cover - import-time environment guard
    raise RuntimeError(
        "This pipeline requires pymatgen. Install requirements.txt before training."
    ) from exc


ALL_ELEMENTS: list[Element] = []
for z in range(1, 119):
    try:
        ALL_ELEMENTS.append(Element.from_Z(z))
    except Exception:
        pass

ELEMENT_SYMBOLS = [el.symbol for el in ALL_ELEMENTS]
BLOCKS = ("s", "p", "d", "f")
PERIODS = tuple(range(1, 8))
GROUPS = tuple(range(1, 19))
RARE_EARTHS = {
    "Sc",
    "Y",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
}
ALKALI = {"Li", "Na", "K", "Rb", "Cs", "Fr"}
ALKALINE_EARTH = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}
HALOGENS = {"F", "Cl", "Br", "I", "At", "Ts"}
CHALCOGENS = {"O", "S", "Se", "Te", "Po", "Lv"}
PNICTOGENS = {"N", "P", "As", "Sb", "Bi", "Mc"}
TRANSITION_METALS = {
    el.symbol
    for el in ALL_ELEMENTS
    if getattr(el, "block", None) == "d"
}

ELEMENT_PROPERTIES = [
    ("Z", lambda el: el.Z),
    ("atomic_mass", lambda el: el.atomic_mass),
    ("X", lambda el: el.X),
    ("atomic_radius", lambda el: el.atomic_radius),
    ("average_ionic_radius", lambda el: el.average_ionic_radius),
    ("mendeleev_no", lambda el: el.mendeleev_no),
    ("row", lambda el: el.row),
    ("group", lambda el: el.group),
    ("melting_point", lambda el: el.melting_point),
    ("boiling_point", lambda el: el.boiling_point),
    ("density_of_solid", lambda el: el.density_of_solid),
    ("electrical_resistivity", lambda el: el.electrical_resistivity),
    ("thermal_conductivity", lambda el: el.thermal_conductivity),
]


def safe_float(value: object) -> float:
    try:
        if value is None:
            return np.nan
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except Exception:
        return np.nan


def weighted_stats(values: Iterable[float], weights: Iterable[float]) -> dict[str, float]:
    values_arr = np.asarray([safe_float(v) for v in values], dtype=float)
    weights_arr = np.asarray([safe_float(w) for w in weights], dtype=float)
    mask = np.isfinite(values_arr) & np.isfinite(weights_arr) & (weights_arr > 0)
    if not np.any(mask):
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
        }
    values_arr = values_arr[mask]
    weights_arr = weights_arr[mask]
    weights_arr = weights_arr / weights_arr.sum()
    mean = float(np.sum(values_arr * weights_arr))
    variance = float(np.sum(weights_arr * (values_arr - mean) ** 2))
    return {
        "mean": mean,
        "std": math.sqrt(max(variance, 0.0)),
        "min": float(values_arr.min()),
        "max": float(values_arr.max()),
        "range": float(values_arr.max() - values_arr.min()),
    }


def parse_composition(formula: str | None, structure: Structure | None = None) -> Composition | None:
    if isinstance(formula, str) and formula.strip():
        try:
            return Composition(formula)
        except Exception:
            pass
    if structure is not None:
        try:
            return structure.composition
        except Exception:
            return None
    return None


def resolve_cif_path(csv_path: str, split_dir: str | Path, train_test_label: str | None = None) -> str:
    if not isinstance(csv_path, str) or not csv_path:
        return csv_path
    raw = Path(csv_path)
    if raw.exists():
        return str(raw)
    split_dir = Path(split_dir)
    filename = raw.name
    if train_test_label:
        candidate = split_dir / str(train_test_label).strip().lower() / filename
        if candidate.exists():
            return str(candidate)
    for split in ("train", "test", "unlabeled"):
        candidate = split_dir / split / filename
        if candidate.exists():
            return str(candidate)
    return str(raw)


def load_merged_dataset(csv_path: str | Path, split_cif_dir: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    required = {"cif", "formula", "label", "train_test_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["resolved_cif"] = df.apply(
        lambda row: resolve_cif_path(
            row["cif"],
            split_cif_dir,
            row.get("train_test_label"),
        ),
        axis=1,
    )
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    if "Tc" in df.columns:
        df["Tc"] = pd.to_numeric(df["Tc"], errors="coerce")
    return df


def split_train_test(
    df: pd.DataFrame,
    include_unlabeled_in_train: bool = False,
    year_cutoff: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Temporal split: train on materials discovered before the cutoff year,
    # test on those from the cutoff year onward. Falls back to nothing else;
    # used only when year_cutoff is provided and a 'year' column exists.
    if year_cutoff is not None:
        if "year" not in df.columns:
            raise ValueError(
                "year_cutoff requested but no 'year' column in the CSV. "
                "Use a CSV that includes a 'year' column."
            )
        years = pd.to_numeric(df["year"], errors="coerce")
        train_mask = years < year_cutoff
        test_mask = years >= year_cutoff
        return df.loc[train_mask].copy(), df.loc[test_mask].copy()

    split = df["train_test_label"].astype(str).str.lower()
    train_mask = split == "train"
    if include_unlabeled_in_train:
        train_mask = train_mask | (split == "unlabeled")
    test_mask = split == "test"
    return df.loc[train_mask].copy(), df.loc[test_mask].copy()


def composition_features(comp: Composition | None) -> dict[str, float]:
    features: dict[str, float] = {}
    if comp is None:
        return features

    el_amt = comp.get_el_amt_dict()
    total = float(sum(el_amt.values()))
    if total <= 0:
        return features

    fractions = {sym: amt / total for sym, amt in el_amt.items()}
    frac_values = np.asarray(list(fractions.values()), dtype=float)

    features["comp_num_elements"] = float(len(fractions))
    features["comp_total_atoms"] = total
    features["comp_reduced_atoms"] = float(comp.reduced_composition.num_atoms)
    features["comp_max_fraction"] = float(frac_values.max())
    features["comp_min_fraction"] = float(frac_values.min())
    features["comp_fraction_std"] = float(frac_values.std())
    features["comp_fraction_l2"] = float(np.sqrt(np.sum(frac_values**2)))
    features["comp_entropy"] = float(-np.sum(frac_values * np.log(frac_values + 1e-12)))
    features["comp_entropy_norm"] = features["comp_entropy"] / math.log(len(frac_values) + 1e-12)
    features["comp_trace_dopant_fraction"] = float(np.sum(frac_values[frac_values < 0.05]))
    features["comp_num_trace_elements"] = float(np.sum(frac_values < 0.05))

    for sym in ELEMENT_SYMBOLS:
        features[f"el_frac_{sym}"] = float(fractions.get(sym, 0.0))

    for block in BLOCKS:
        features[f"block_frac_{block}"] = 0.0
    for period in PERIODS:
        features[f"period_frac_{period}"] = 0.0
    for group in GROUPS:
        features[f"group_frac_{group}"] = 0.0

    elements: list[Element] = []
    weights: list[float] = []
    for sym, frac in fractions.items():
        try:
            el = Element(sym)
        except Exception:
            continue
        elements.append(el)
        weights.append(frac)
        if el.block in BLOCKS:
            features[f"block_frac_{el.block}"] += frac
        if el.row in PERIODS:
            features[f"period_frac_{el.row}"] += frac
        if el.group in GROUPS:
            features[f"group_frac_{el.group}"] += frac

    symbols = set(fractions)
    features["family_oxide"] = float("O" in symbols)
    features["family_cuprate_signal"] = float("Cu" in symbols and "O" in symbols)
    features["family_fe_pnictide_signal"] = float("Fe" in symbols and bool(symbols & PNICTOGENS))
    features["family_fe_chalcogenide_signal"] = float("Fe" in symbols and bool(symbols & CHALCOGENS))
    features["family_mgb2_signal"] = float("Mg" in symbols and "B" in symbols)
    features["family_a15_signal"] = float(("Nb" in symbols or "V" in symbols) and len(symbols) <= 3)
    features["family_hydride"] = float("H" in symbols)
    features["frac_rare_earth"] = float(sum(fractions.get(sym, 0.0) for sym in RARE_EARTHS))
    features["frac_transition_metal"] = float(sum(fractions.get(sym, 0.0) for sym in TRANSITION_METALS))
    features["frac_alkali"] = float(sum(fractions.get(sym, 0.0) for sym in ALKALI))
    features["frac_alkaline_earth"] = float(sum(fractions.get(sym, 0.0) for sym in ALKALINE_EARTH))
    features["frac_halogen"] = float(sum(fractions.get(sym, 0.0) for sym in HALOGENS))
    features["frac_chalcogen"] = float(sum(fractions.get(sym, 0.0) for sym in CHALCOGENS))
    features["frac_pnictogen"] = float(sum(fractions.get(sym, 0.0) for sym in PNICTOGENS))
    features["pair_frac_Cu_O"] = float(fractions.get("Cu", 0.0) * fractions.get("O", 0.0))
    features["pair_frac_Fe_As"] = float(fractions.get("Fe", 0.0) * fractions.get("As", 0.0))
    features["pair_frac_Fe_SeTe"] = float(fractions.get("Fe", 0.0) * (fractions.get("Se", 0.0) + fractions.get("Te", 0.0)))
    features["pair_frac_Nb_Sn"] = float(fractions.get("Nb", 0.0) * fractions.get("Sn", 0.0))
    features["pair_frac_Mg_B"] = float(fractions.get("Mg", 0.0) * fractions.get("B", 0.0))

    for prop_name, getter in ELEMENT_PROPERTIES:
        values = [safe_float(getter(el)) for el in elements]
        stats = weighted_stats(values, weights)
        for stat_name, value in stats.items():
            features[f"prop_{prop_name}_{stat_name}"] = value

    return features


def structural_features(structure: Structure | None) -> dict[str, float]:
    features: dict[str, float] = {}
    if structure is None:
        return features

    try:
        lattice = structure.lattice
        abc = np.asarray(lattice.abc, dtype=float)
        angles = np.asarray(lattice.angles, dtype=float)
        features["struct_num_sites"] = float(len(structure))
        features["struct_volume"] = float(structure.volume)
        features["struct_density"] = float(structure.density)
        features["struct_volume_per_site"] = float(structure.volume / max(len(structure), 1))
        features["struct_lattice_a"] = float(abc[0])
        features["struct_lattice_b"] = float(abc[1])
        features["struct_lattice_c"] = float(abc[2])
        features["struct_lattice_alpha"] = float(angles[0])
        features["struct_lattice_beta"] = float(angles[1])
        features["struct_lattice_gamma"] = float(angles[2])
        sorted_abc = np.sort(abc)
        features["struct_aspect_max_min"] = float(sorted_abc[-1] / max(sorted_abc[0], 1e-12))
        features["struct_aspect_mid_min"] = float(sorted_abc[1] / max(sorted_abc[0], 1e-12))
        features["struct_is_ordered"] = float(structure.is_ordered)
        features["struct_disordered_site_fraction"] = float(
            np.mean([not site.is_ordered for site in structure.sites])
        )
    except Exception:
        pass

    try:
        _symbol, number = structure.get_space_group_info(symprec=0.1)
        features["struct_space_group_number"] = float(number)
    except Exception:
        features["struct_space_group_number"] = np.nan

    try:
        if 1 < len(structure) <= 250:
            distances = np.asarray(structure.distance_matrix, dtype=float)
            distances = distances[np.triu_indices_from(distances, k=1)]
            distances = distances[np.isfinite(distances) & (distances > 1e-8)]
            if distances.size:
                features["dist_min"] = float(np.min(distances))
                features["dist_mean"] = float(np.mean(distances))
                features["dist_std"] = float(np.std(distances))
                features["dist_p10"] = float(np.percentile(distances, 10))
                features["dist_p50"] = float(np.percentile(distances, 50))
                features["dist_p90"] = float(np.percentile(distances, 90))
                hist, _ = np.histogram(distances, bins=np.linspace(0.0, 10.0, 21))
                hist = hist.astype(float) / max(float(hist.sum()), 1.0)
                for idx, value in enumerate(hist):
                    features[f"rdf_hist_{idx:02d}"] = float(value)
    except Exception:
        pass

    return features


def read_structure(path: str | Path) -> Structure | None:
    try:
        if not path or not os.path.exists(path):
            return None
        return Structure.from_file(str(path))
    except Exception:
        return None


def featurize_row(row: pd.Series) -> dict[str, float]:
    structure = read_structure(row.get("resolved_cif") or row.get("cif"))
    comp = parse_composition(row.get("formula"), structure=structure)
    features = composition_features(comp)
    features.update(structural_features(structure))
    features["feature_has_structure"] = float(structure is not None)
    features["feature_has_composition"] = float(comp is not None)
    return features


def build_feature_frame(df: pd.DataFrame, cache_path: str | Path | None = None) -> pd.DataFrame:
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists():
        cached = pd.read_pickle(cache)
        if len(cached) == len(df):
            return cached

    records = []
    for _, row in df.iterrows():
        records.append(featurize_row(row))
    features = pd.DataFrame(records, index=df.index)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.reindex(sorted(features.columns), axis=1)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        features.to_pickle(cache)
    return features
