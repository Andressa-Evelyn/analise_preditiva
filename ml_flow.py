import mlflow
import mlflow.sklearn
import kagglehub
from kagglehub import KaggleDatasetAdapter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

# =========================
# MLflow config
# =========================
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_id=1)

# =========================
# Load dataset
# =========================
file_path = "Dropout_Academic Success - Sheet1.csv"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "ankanhore545/dropout-or-academic-success",
    file_path,
)

# =========================
# Features / Target
# =========================
X = df.drop(columns=["Target"])
y = df["Target"]

# =========================
# Encode target
# =========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# =========================
# Scale features
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# Train / Validation split
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =========================
# MLflow run
# =========================
with mlflow.start_run():

    # Autolog funciona PERFEITAMENTE com sklearn
    mlflow.sklearn.autolog()

    # =========================
    # Model
    # =========================
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=3,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # =========================
    # Train
    # =========================
    model.fit(X_train, y_train)

    # =========================
    # Predict
    # =========================
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    # =========================
    # Threshold para Dropout
    # =========================
    DROPOUT_CLASS = 0
    threshold = 0.30

    dropout_probs = y_proba[:, DROPOUT_CLASS]
    y_pred_thresh = np.argmax(y_proba, axis=1)
    y_pred_thresh[dropout_probs >= threshold] = DROPOUT_CLASS

    # =========================
    # Metrics
    # =========================
    precision, recall, f1, support = precision_recall_fscore_support(
        y_val,
        y_pred_thresh,
        labels=np.unique(y_val)
    )

    # =========================
    # Log metrics manual (por classe)
    # =========================
    for i, class_name in enumerate(le.classes_):
        mlflow.log_metric(f"Precision_{class_name}", float(precision[i]))
        mlflow.log_metric(f"Recall_{class_name}", float(recall[i]))
        mlflow.log_metric(f"F1_{class_name}", float(f1[i]))

    mlflow.log_metric("F1_macro", float(np.mean(f1)))

    # =========================
    # Log label encoder + scaler
    # =========================
    mlflow.log_dict(
        {"classes": le.classes_.tolist()},
        artifact_file="label_encoder.json"
    )
