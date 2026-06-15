import argparse
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.core import Composition, Element
from pymatgen.io.cif import CifParser
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --- Featurization ---
def featurize_cif(path: str) -> dict | None:
    """
    Parses a CIF file and extracts a richer set of compositional and basic structural features.

    Args:
        path (str): Absolute path to the CIF file.

    Returns:
        dict | None: A dictionary of features if parsing is successful, None otherwise.
    """
    try:
        parser = CifParser(path)
        structures = parser.get_structures(primitive=False)
        if not structures:
            return None  # No valid structure found in CIF
        structure = structures[0]  # Take the first structure
        comp = structure.composition

        features = {}
        features["num_elements"] = len(comp.elements)
        features["total_atomic_mass"] = comp.weight  # Total weight of the formula unit

        # Add basic structural features
        features["volume"] = structure.volume
        features["density_crystal"] = (
            structure.density
        )  # Renamed to avoid confusion with elemental density average
        features["num_sites"] = structure.num_sites
        try:
            sg_analyzer = SpacegroupAnalyzer(structure)
            features["space_group_number"] = sg_analyzer.get_space_group_number()
        except Exception:
            features["space_group_number"] = (
                np.nan
            )  # Handle cases where space group analysis fails

        # List of elemental properties to calculate weighted averages for
        props_to_avg = [
            "atomic_radius",
            "X",  # Electronegativity (Pauling)
            "melting_point",
            "boiling_point",
            "density",  # Elemental density, distinct from structure density
            "n_valence_electrons",
            "mendeleev_number",
            "atomic_mass",
            "coefficient_of_linear_thermal_expansion",
            "electrical_resistivity",
            "thermal_conductivity",
            "sound_speed",
            "electron_affinity",
            "ionization_energy",
        ]

        # Properties for which min, max, std deviation will be calculated
        # Removed 'range' from the intended calculations for these properties
        props_for_stats = [
            "atomic_radius",
            "X",
            "melting_point",
            "n_valence_electrons",
            "atomic_mass",
            "ionization_energy",
        ]

        for prop_name in props_to_avg:
            weighted_sum = 0
            total_fraction = 0
            prop_values_for_stats = []  # Store values for min/max/std calculation

            for el in comp.elements:
                fraction = comp.get_atomic_fraction(el)
                prop_val = getattr(el, prop_name, None)

                # Ensure prop_val is not None and not NaN before using
                if prop_val is not None and not pd.isna(prop_val):
                    weighted_sum += fraction * prop_val
                    total_fraction += fraction
                    if prop_name in props_for_stats:
                        prop_values_for_stats.append(prop_val)

            # Weighted average
            features[f"{prop_name}_avg"] = (
                weighted_sum / total_fraction if total_fraction > 0 else np.nan
            )

            # Min, Max, and Std for selected properties (removed range)
            if prop_name in props_for_stats and len(prop_values_for_stats) > 0:
                features[f"{prop_name}_min"] = np.min(prop_values_for_stats)
                features[f"{prop_name}_max"] = np.max(prop_values_for_stats)
                features[f"{prop_name}_std"] = np.std(prop_values_for_stats)
                # Removed: features[f"{prop_name}_range"] = np.max(prop_values_for_stats) - np.min(prop_values_for_stats)
            else:
                features[f"{prop_name}_min"] = np.nan
                features[f"{prop_name}_max"] = np.nan
                features[f"{prop_name}_std"] = np.nan
                # Removed: features[f"{prop_name}_range"] = np.nan

        return features
    except Exception:
        # Catch any parsing or featurization errors and return None
        return None


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Builds a feature matrix (X) and target vector (y) from the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'cif' paths and 'Tc' values.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple of (features_df, target_series).
    """
    features_list = []
    processed_indices = []
    total_rows = len(df)
    print(f"Featurizing {total_rows} CIF files...")
    progress_interval = max(1, total_rows // 20) if total_rows else 1

    # Iterate through each row and featurize CIF files
    for processed_count, (idx, row) in enumerate(df.iterrows(), start=1):
        cif_path = row["cif"]
        if (
            processed_count == 1
            or processed_count == total_rows
            or processed_count % progress_interval == 0
        ):
            print(f"  featurized {processed_count}/{total_rows}")
        feats = featurize_cif(cif_path)
        if feats is not None:
            features_list.append(feats)
            processed_indices.append(idx)  # Store original index for alignment

    if not features_list:
        # Return empty dataframes if no features could be extracted
        return pd.DataFrame(), pd.Series()

    # Create a DataFrame from the extracted features, using original indices
    X = pd.DataFrame(features_list, index=processed_indices)

    # Align target variable 'Tc' with the successfully featurized samples
    y = df.loc[processed_indices, "Tc"]

    # --- Preprocessing/Noise Suppression ---
    # Impute missing feature values (NaNs) using the median strategy.
    # This handles cases where some elemental properties are not available for all elements
    # or if total_fraction was 0 in featurize_cif.
    # It also handles newly added structural features if they fail to be extracted.
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            # If median_val is NaN itself (e.g., column is all NaNs), impute with 0
            X[col] = X[col].fillna(median_val if not pd.isna(median_val) else 0)

    # Final check for any remaining NaNs (e.g., if a column was entirely NaN and median_val was NaN)
    X = X.fillna(0)  # Fallback to 0 if any NaNs somehow persist

    print(f"Built feature matrix with shape {X.shape}")
    print(f"Successfully featurized {len(processed_indices)} of {total_rows} rows")

    return X, y


# --- Model Training and Prediction ---
def train_model(X: pd.DataFrame, y: pd.Series):
    """
    Trains an XGBoost Regressor model.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.

    Returns:
        xgb.XGBRegressor: Trained XGBoost model.
    """
    # Initialize XGBoost Regressor with reasonable parameters for performance and speed
    model = xgb.XGBRegressor(
        objective="reg:absoluteerror",  # Align objective with MAE evaluation metric
        n_estimators=2500,  # Number of boosting rounds
        learning_rate=0.1,  # Step size shrinkage to prevent overfitting
        max_depth=6,  # Maximum depth of a tree
        subsample=0.8,  # Subsample ratio of the training instance
        colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
        random_state=42,  # For reproducibility
        n_jobs=-1,  # Use all available CPU cores
        tree_method="hist",  # Faster tree construction for larger datasets
    )

    print(f"Training regressor on {len(y)} samples with {X.shape[1]} features...")

    # Fit the model to the entire provided training data
    model.fit(X, y)
    print("Model training complete")
    return model


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """
    Makes predictions using the trained model.

    Args:
        model: Trained model (e.g., xgb.XGBRegressor).
        X (pd.DataFrame): Feature matrix for prediction.

    Returns:
        np.ndarray: Array of predictions.
    """
    return model.predict(X)


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
    Loads the merged dataset and resolves local CIF paths.
    """
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path).copy()
    print(f"Loaded {len(df)} rows")

    required_columns = {"cif", "Tc"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

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


def drop_invalid_targets(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Drops rows with non-finite Tc values and reports how many were removed.
    """
    df = df.copy()
    original_count = len(df)
    df["Tc"] = pd.to_numeric(df["Tc"], errors="coerce")
    valid_mask = np.isfinite(df["Tc"])
    removed_count = int((~valid_mask).sum())
    if removed_count:
        print(f"Dropping {removed_count} {dataset_name} rows with invalid Tc values")
    filtered_df = df.loc[valid_mask].copy()
    print(f"{dataset_name.capitalize()} rows after Tc filtering: {len(filtered_df)} / {original_count}")
    return filtered_df


def main():
    parser = argparse.ArgumentParser(
        description="Train an XGBoost regression model on merged_superconductor_data.csv."
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
        default="xgb_superconductor_regressor.joblib",
        help="Path to save the trained model",
    )
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    split_cif_dir = os.path.abspath(args.split_cif_dir)
    output_model = os.path.abspath(args.output_model)

    print("Starting regression training run")
    df = load_dataset(csv_path=csv_path, split_dir=split_cif_dir)

    if "train_test_label" in df.columns:
        train_df = df[df["train_test_label"].astype(str).str.lower() == "train"].copy()
        test_df = df[df["train_test_label"].astype(str).str.lower() == "test"].copy()
    else:
        train_df = df.copy()
        test_df = pd.DataFrame(columns=df.columns)

    train_df = drop_invalid_targets(train_df, "training")
    if not test_df.empty:
        test_df = drop_invalid_targets(test_df, "test")

    if train_df.empty:
        raise ValueError("No training rows found in the dataset.")

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    print("Building training features...")
    X_train, y_train = build_features(train_df)
    if X_train.empty:
        raise ValueError("No training features could be extracted from the CIF files.")

    print("Training model...")
    model = train_model(X_train, y_train)

    os.makedirs(os.path.dirname(output_model) or ".", exist_ok=True)
    print(f"Saving model to: {output_model}")
    joblib.dump(model, output_model)
    print(f"Saved model to: {output_model}")

    if not test_df.empty:
        print("Building test features...")
        X_test, y_test = build_features(test_df)
        if not X_test.empty:
            print("Running evaluation...")
            y_pred = predict(model, X_test)
            results = test_df.loc[X_test.index].copy()

            results["actual_Tc"] = y_test.values
            results["predicted_Tc"] = y_pred

            results["relative_error"] = (
                abs(results["actual_Tc"] - results["predicted_Tc"])
                / abs(results["actual_Tc"])
            )

            high_error = results[
                results["relative_error"] > 0.50
            ]

            high_error.to_csv(
                "regression_high_relative_error_1987.csv",
                index=False
            )

            print(
                f"Saved {len(high_error)} high relative error materials"
            )
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            print(f"Test rows used: {len(X_test)}")
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R2: {r2:.4f}")
        else:
            print("No test features could be extracted; skipping evaluation.")

    print("Regression training run finished")


if __name__ == "__main__":
    main()
