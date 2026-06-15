import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Element
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Suppress specific warnings from pymatgen or other libraries
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")


# Define a template for all possible features with np.nan values
# This ensures that even if featurization completely fails, a dictionary
# with all expected keys is returned, preventing issues with DataFrame creation.
_FEATURE_TEMPLATE = {
    "num_sites": np.nan,
    "density": np.nan,
    "volume": np.nan,
    "lattice_a": np.nan,
    "lattice_b": np.nan,
    "lattice_c": np.nan,
    "lattice_alpha": np.nan,
    "lattice_beta": np.nan,
    "lattice_gamma": np.nan,
    "num_elements": np.nan,
    "avg_atomic_mass": np.nan,
    "std_atomic_mass": np.nan,
    "avg_electronegativity": np.nan,
    "max_electronegativity": np.nan,
    "min_electronegativity": np.nan,
    "std_electronegativity": np.nan,
    "avg_atomic_radius": np.nan,
    "max_atomic_radius": np.nan,
    "min_atomic_radius": np.nan,
    "std_atomic_radius": np.nan,
    "avg_group": np.nan,
    "max_group": np.nan,
    "min_group": np.nan,
    "std_group": np.nan,
    # NEW FEATURES: Row statistics
    "avg_row": np.nan,
    "max_row": np.nan,
    "min_row": np.nan,
    "std_row": np.nan,
    "s_block_fraction": np.nan,
    "p_block_fraction": np.nan,
    "d_block_fraction": np.nan,
    "f_block_fraction": np.nan,
    "weighted_avg_atomic_mass": np.nan,
    "weighted_std_atomic_mass": np.nan,
    "weighted_avg_electronegativity": np.nan,
    "weighted_std_electronegativity": np.nan,
    "weighted_avg_atomic_radius": np.nan,
    "weighted_std_atomic_radius": np.nan,
    "weighted_avg_group": np.nan,
    "weighted_std_group": np.nan,
    # NEW FEATURES: Weighted Row statistics
    "weighted_avg_row": np.nan,
    "weighted_std_row": np.nan,
    "space_group_number": np.nan,
}


def featurize_cif(path: str) -> dict:
    """
    Featurizes a CIF file into a dictionary of features.
    Handles malformed/unreadable CIFs by returning a dictionary
    with np.nan for all features.
    """
    features = _FEATURE_TEMPLATE.copy()  # Start with a fresh template

    try:
        # Ensure the path exists
        if not os.path.exists(path):
            return features  # Return template with NaNs if file doesn't exist

        structure = Structure.from_file(path)

        # Basic structural properties
        features["num_sites"] = len(structure)
        features["density"] = structure.density
        features["volume"] = structure.volume

        # Lattice parameters
        features["lattice_a"] = structure.lattice.a
        features["lattice_b"] = structure.lattice.b
        features["lattice_c"] = structure.lattice.c
        features["lattice_alpha"] = structure.lattice.alpha
        features["lattice_beta"] = structure.lattice.beta
        features["lattice_gamma"] = structure.lattice.gamma

        # Compositional properties
        composition = structure.composition
        elements = list(composition.elements)  # List of unique elements
        features["num_elements"] = len(elements)

        if not elements:
            # If no elements, return current features (which will have NaNs for compositional stats)
            return features

        # Collect properties for aggregation, handling potential None values
        atomic_masses = [e.atomic_mass for e in elements if e.atomic_mass is not None]
        electronegativities = [e.X for e in elements if e.X is not None]
        atomic_radii = [
            e.atomic_radius for e in elements if e.atomic_radius is not None
        ]
        groups = [e.group for e in elements if e.group is not None]
        rows = [e.row for e in elements if e.row is not None]  # Collect row values

        # Aggregate properties, providing np.nan for empty lists
        features["avg_atomic_mass"] = (
            np.mean(atomic_masses) if atomic_masses else np.nan
        )
        features["std_atomic_mass"] = np.std(atomic_masses) if atomic_masses else np.nan
        features["avg_electronegativity"] = (
            np.mean(electronegativities) if electronegativities else np.nan
        )
        features["max_electronegativity"] = (
            np.max(electronegativities) if electronegativities else np.nan
        )
        features["min_electronegativity"] = (
            np.min(electronegativities) if electronegativities else np.nan
        )
        features["std_electronegativity"] = (
            np.std(electronegativities) if electronegativities else np.nan
        )
        features["avg_atomic_radius"] = (
            np.mean(atomic_radii) if atomic_radii else np.nan
        )
        features["max_atomic_radius"] = np.max(atomic_radii) if atomic_radii else np.nan
        features["min_atomic_radius"] = np.min(atomic_radii) if atomic_radii else np.nan
        features["std_atomic_radius"] = np.std(atomic_radii) if atomic_radii else np.nan
        features["avg_group"] = np.mean(groups) if groups else np.nan
        features["max_group"] = np.max(groups) if groups else np.nan
        features["min_group"] = np.min(groups) if groups else np.nan
        features["std_group"] = np.std(groups) if groups else np.nan

        # NEW: Row statistics
        features["avg_row"] = np.mean(rows) if rows else np.nan
        features["max_row"] = np.max(rows) if rows else np.nan
        features["min_row"] = np.min(rows) if rows else np.nan
        features["std_row"] = np.std(rows) if rows else np.nan

        # Block fractions (now based on atomic fractions)
        s_block_fraction_sum = 0.0
        p_block_fraction_sum = 0.0
        d_block_fraction_sum = 0.0
        f_block_fraction_sum = 0.0

        total_atoms = composition.num_atoms
        if total_atoms > 0:
            for el, amount in composition.items():
                if el.block == "s":
                    s_block_fraction_sum += amount
                elif el.block == "p":
                    p_block_fraction_sum += amount
                elif el.block == "d":
                    d_block_fraction_sum += amount
                elif el.block == "f":
                    f_block_fraction_sum += amount

            features["s_block_fraction"] = s_block_fraction_sum / total_atoms
            features["p_block_fraction"] = p_block_fraction_sum / total_atoms
            features["d_block_fraction"] = d_block_fraction_sum / total_atoms
            features["f_block_fraction"] = f_block_fraction_sum / total_atoms
        else:
            features["s_block_fraction"] = np.nan
            features["p_block_fraction"] = np.nan
            features["d_block_fraction"] = np.nan
            features["f_block_fraction"] = np.nan

        # Weighted Compositional Statistics
        def calculate_weighted_stats(data_list):
            """
            Calculates weighted mean and weighted standard deviation.
            data_list: list of (value, weight) tuples.
            """
            if not data_list:
                return np.nan, np.nan

            values = np.array([d[0] for d in data_list])
            weights = np.array([d[1] for d in data_list])

            total_weight = np.sum(weights)
            if total_weight == 0:
                return np.nan, np.nan  # Avoid division by zero if all weights are zero

            weighted_mean = np.sum(values * weights) / total_weight

            # Calculate weighted variance: sum(w_i * (x_i - mean)^2) / sum(w_i)
            weighted_variance = (
                np.sum(weights * (values - weighted_mean) ** 2) / total_weight
            )
            weighted_std = np.sqrt(weighted_variance)

            return weighted_mean, weighted_std

        # Collect data for weighted calculations, filtering out None properties
        atomic_mass_data = []  # list of (value, amount)
        electronegativity_data = []
        atomic_radius_data = []
        group_data = []
        row_data = []  # NEW: Collect row data for weighted stats

        for el, amount in composition.items():
            if el.atomic_mass is not None:
                atomic_mass_data.append((el.atomic_mass, amount))
            if el.X is not None:
                electronegativity_data.append((el.X, amount))
            if el.atomic_radius is not None:
                atomic_radius_data.append((el.atomic_radius, amount))
            if el.group is not None:
                group_data.append((el.group, amount))
            if el.row is not None:  # NEW: Add row to weighted stats
                row_data.append((el.row, amount))

        # Add new weighted features
        features["weighted_avg_atomic_mass"], features["weighted_std_atomic_mass"] = (
            calculate_weighted_stats(atomic_mass_data)
        )
        (
            features["weighted_avg_electronegativity"],
            features["weighted_std_electronegativity"],
        ) = calculate_weighted_stats(electronegativity_data)
        (
            features["weighted_avg_atomic_radius"],
            features["weighted_std_atomic_radius"],
        ) = calculate_weighted_stats(atomic_radius_data)
        features["weighted_avg_group"], features["weighted_std_group"] = (
            calculate_weighted_stats(group_data)
        )
        # NEW: Add weighted row features
        features["weighted_avg_row"], features["weighted_std_row"] = (
            calculate_weighted_stats(row_data)
        )

        # Space Group Number
        try:
            # get_space_group_info returns (symbol, number, point_group)
            features["space_group_number"] = structure.get_space_group_info()[1]
        except Exception:
            # If space group info cannot be retrieved for any reason, set to NaN
            features["space_group_number"] = np.nan

        return features

    except Exception:
        # Catch any exceptions during parsing or feature extraction
        # and return the template with all features set to np.nan
        return features


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Builds raw features from the input DataFrame containing CIF paths.
    It applies featurization and returns a DataFrame with potential np.nan values,
    without any imputation or scaling, to prevent data leakage.
    """
    df_copy = df.copy()  # Work on a copy to avoid modifying original df
    total_rows = len(df_copy)
    print(f"Featurizing {total_rows} CIF files...")

    features = []
    progress_interval = max(1, total_rows // 20)

    for idx, cif_path in enumerate(df_copy["cif"], start=1):
        if idx == 1 or idx == total_rows or idx % progress_interval == 0:
            print(f"  featurized {idx}/{total_rows}")
        features.append(featurize_cif(cif_path))

    df_copy["features"] = features

    # Expand features dictionary into separate columns
    X_raw = pd.DataFrame(df_copy["features"].tolist())
    y = df_copy["class"]

    print(f"Built feature matrix with shape {X_raw.shape}")

    # Return raw features and target, preprocessing will be handled in train_model
    return X_raw, y


def train_model(X_raw: pd.DataFrame, y: pd.Series):
    """
    Trains an XGBoost classifier model within a scikit-learn pipeline
    that includes preprocessing steps (imputation and scaling).
    This prevents data leakage by fitting the preprocessor only on the training data.
    """
    print("Preparing preprocessing pipeline...")

    # Identify numerical features for imputation and scaling
    numerical_features = X_raw.select_dtypes(include=np.number).columns.tolist()
    print(f"Detected {len(numerical_features)} numerical features")

    # Create a preprocessing pipeline for numerical features
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    # Create a ColumnTransformer to apply preprocessing to numerical features
    preprocessor = ColumnTransformer(
        transformers=[("num", numerical_transformer, numerical_features)],
        remainder="passthrough",  # Keep other columns (if any, though none expected here)
    )

    # Determine number of classes for XGBoost
    num_classes = y.nunique()
    print(f"Training classifier with {num_classes} classes on {len(y)} samples")

    # Initialize XGBoost Classifier
    xgb_classifier = XGBClassifier(
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_estimators=200,
        learning_rate=0.1,
        random_state=42,
        tree_method="hist",
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
    )

    # Create the full pipeline: preprocessor + classifier
    model_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", xgb_classifier)]
    )

    print("Fitting model...")

    # Fit the entire pipeline on the raw training data
    model_pipeline.fit(X_raw, y)
    print("Model training complete")
    return model_pipeline


def predict(model: Pipeline, X_raw: pd.DataFrame) -> np.ndarray:
    """
    Makes predictions using the trained pipeline.
    The pipeline handles preprocessing of raw features before prediction.
    """
    predictions = model.predict(X_raw)
    return predictions


def resolve_local_cif_path(
    original_path: str,
    split_dir: str,
    train_test_label: str | None = None,
) -> str:
    """
    Maps the CIF path stored in the CSV to a local CIF file in split_cifs_by_csv_label.
    """
    if not isinstance(original_path, str) or not original_path:
        return original_path

    filename = os.path.basename(original_path)
    if train_test_label:
        subset = str(train_test_label).strip().lower()
        labeled_candidate = os.path.join(split_dir, subset, filename)
        if os.path.exists(labeled_candidate):
            return labeled_candidate

    for subset in ("train", "test"):
        split_candidate = os.path.join(split_dir, subset, filename)
        if os.path.exists(split_candidate):
            return split_candidate

    return original_path


def load_dataset(csv_path: str, split_dir: str) -> pd.DataFrame:
    """
    Loads the merged dataset and adapts its columns to the exact training script.
    """
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path).copy()
    print(f"Loaded {len(df)} rows")

    required_columns = {"label", "cif"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    df = df.rename(columns={"label": "class"})
    print(f"Resolving CIF paths from: {split_dir}")
    df["cif"] = df.apply(
        lambda row: resolve_local_cif_path(
            row["cif"],
            split_dir=split_dir,
            train_test_label=row.get("train_test_label"),
        ),
        axis=1,
    )
    missing_cifs = (~df["cif"].apply(os.path.exists)).sum()
    print(f"Resolved CIF paths. Missing CIF files: {missing_cifs}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Train an XGBoost classification model on merged_superconductor_data.csv."
    )
    parser.add_argument(
        "--csv",
        default="merged_superconductor_data.csv",
        help="Path to merged_superconductor_data.csv",
    )
    parser.add_argument(
        "--split-cif-dir",
        default="split_cifs_by_csv_label",
        help="Directory containing train/test CIF subdirectories",
    )
    parser.add_argument(
        "--output-model",
        default="xgb_superconductor_classifier.joblib",
        help="Path to save the trained model pipeline",
    )
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    split_cif_dir = os.path.abspath(args.split_cif_dir)
    output_model = os.path.abspath(args.output_model)

    print("Starting training run")
    df = load_dataset(csv_path=csv_path, split_dir=split_cif_dir)

    if "train_test_label" in df.columns:
        train_df = df[df["train_test_label"].astype(str).str.lower() == "train"].copy()
        test_df = df[df["train_test_label"].astype(str).str.lower() == "test"].copy()
    else:
        train_df = df.copy()
        test_df = pd.DataFrame(columns=df.columns)

    if train_df.empty:
        raise ValueError("No training rows found in the dataset.")

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    print("Building training features...")
    X_train_raw, y_train = build_features(train_df)
    print("Training model...")
    model = train_model(X_train_raw, y_train)

    os.makedirs(os.path.dirname(output_model) or ".", exist_ok=True)
    print(f"Saving model to: {output_model}")
    joblib.dump(model, output_model)

    print(f"Training rows: {len(train_df)}")
    print(f"Saved model to: {output_model}")

    if not test_df.empty:
        print("Building test features...")
        X_test_raw, y_test = build_features(test_df)
        print("Running evaluation...")
        y_pred = predict(model, X_test_raw)

        results = test_df.copy()
        results["true_label"] = y_test
        results["predicted_label"] = y_pred

        wrong = results[
            results["true_label"] != results["predicted_label"]
        ]

        wrong.to_csv(
            "classification_wrong_predictions_1987.csv",
            index=False
        )

        print(f"Saved {len(wrong)} wrong predictions")

        print(f"Test rows: {len(test_df)}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1 score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"Test rows: {len(test_df)}")
        print("Classification report:")
        print(classification_report(y_test, y_pred, digits=4))

    print("Training run finished")


if __name__ == "__main__":
    main()
