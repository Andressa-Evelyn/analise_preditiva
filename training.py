import os
import argparse
import pickle
from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier, Pool


RESULTS_DIR = Path(__file__).parent / "notebooks" / "andressa" / "resultados"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_from_pickles(base_dir=Path(__file__).parent / "notebooks" / "andressa" / "resultados"):
    """Try to load preprocessed X and y from pickles saved by the notebook.
    Returns DataFrame X and Series y or (None, None) if not found.
    """
    x_path = base_dir / "X.pkl"
    y_path = base_dir / "y.pkl"
    if x_path.exists() and y_path.exists():
        X = pd.read_pickle(x_path)
        y = pd.read_pickle(y_path)
        return X, y
    return None, None


def load_dataset_via_kaggle(file_path="Dropout_Academic Success - Sheet1.csv"):
    df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "ankanhore545/dropout-or-academic-success", file_path)
    return df


def prepare_data(df, target_col="Target", scale=True, encode=True, test_size=0.2, random_state=42):
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in dataframe")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    le = None
    if encode:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Keep X as DataFrame so we preserve column names for CatBoost Pool
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    scaler = None
    if scale and numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))

    return X_train, X_val, y_train, y_val, scaler, le, class_weight_dict


def train_catboost(X_train, X_val, y_train, y_val, class_weight_dict=None, params=None):
    if params is None:
        params = dict(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            random_seed=42,
            auto_class_weights="Balanced",
            border_count=252,
            iterations=1000,
            learning_rate=0.05,
            depth=8,
            early_stopping_rounds=50,
            verbose=100,
        )

    model = CatBoostClassifier(**params)

    train_pool = Pool(X_train, y_train, feature_names=list(X_train.columns))
    val_pool = Pool(X_val, y_val, feature_names=list(X_val.columns))

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    y_pred = model.predict(X_val)
    try:
        y_proba = model.predict_proba(X_val)
    except Exception:
        y_proba = None

    return model, y_pred, y_proba


def save_artifacts(model, X_val, y_val, y_pred, y_proba, le=None):
    # Save predictions and probabilities
    out_pred = pd.DataFrame({"y_true": y_val, "y_pred": y_pred})
    out_pred_path = RESULTS_DIR / "predictions.csv"
    out_pred.to_csv(out_pred_path, index=False)

    if y_proba is not None:
        mean_probs = np.mean(y_proba, axis=0)
        classes = le.classes_ if le is not None else list(range(len(mean_probs)))
        df_probs = pd.DataFrame({"class": classes, "mean_probability": mean_probs})
        df_probs.to_csv(RESULTS_DIR / "prediction_probabilities_summary.csv", index=False)

    # Save model metrics (classification report as dict)
    try:
        from sklearn.metrics import classification_report

        report = classification_report(y_val, y_pred, output_dict=True)
        df_metrics = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "class"})
        df_metrics.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    except Exception:
        pass

    # Feature importances (if supported)
    try:
        fi = model.get_feature_importance()
        df_fi = pd.DataFrame({"feature": list(X_val.columns), "importance": fi})
        df_fi = df_fi.sort_values("importance", ascending=False)
        df_fi.to_csv(RESULTS_DIR / "feature_importances.csv", index=False)
    except Exception:
        pass


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-pickles", action="store_true", help="Load X.pkl and y.pkl from notebooks/andressa/resultados if present")
    parser.add_argument("--file", type=str, default=None, help="CSV file path to load if pickles not present")
    parsed = parser.parse_args(args=args)

    X = y = None
    if parsed.use_pickles:
        X, y = load_from_pickles()

    df = None
    if X is None or y is None:
        if parsed.file:
            df = pd.read_csv(parsed.file)
        else:
            # fallback: try Kaggle dataset
            df = load_dataset_via_kaggle()

        X_train, X_val, y_train, y_val, scaler, le, class_weight_dict = prepare_data(df)
    else:
        # X and y loaded as DataFrame/Series
        # ensure types
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, (pd.Series, np.ndarray)):
            y_ser = pd.Series(y)
        else:
            y_ser = pd.Series(y)
        X_train, X_val, y_train, y_val, scaler, le, class_weight_dict = prepare_data(pd.concat([X, y_ser.rename("Target")], axis=1))

    model, y_pred, y_proba = train_catboost(X_train, X_val, y_train, y_val, class_weight_dict=class_weight_dict)

    save_artifacts(model, X_val, y_val, y_pred, y_proba, le=le)

    print("Training complete. Artifacts saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()