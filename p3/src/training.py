import json
import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler


C_VALUES = [0.001, 0.01, 0.1, 1, 10]


def temporal_split(
    df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[pd.Series, pd.Series, pd.Series]:
    orders_by_date = (
        df[["order_id", "order_date"]]
        .drop_duplicates()
        .sort_values("order_date")
        .reset_index(drop=True)
    )
    n = len(orders_by_date)
    cutoff_val = orders_by_date.iloc[int(n * train_frac)]["order_date"]
    cutoff_test = orders_by_date.iloc[int(n * (train_frac + val_frac))]["order_date"]

    train_mask = df["order_date"] < cutoff_val
    val_mask = (df["order_date"] >= cutoff_val) & (df["order_date"] < cutoff_test)
    test_mask = df["order_date"] >= cutoff_test
    return train_mask, val_mask, test_mask


def train_and_select(
    X_train_s, y_train, X_val_s, y_val
) -> tuple[LogisticRegression, float, float]:
    best_ap, best_model, best_c = -1.0, None, None
    for C in C_VALUES:
        model = LogisticRegression(
            penalty="l2", C=C, solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=42,
        )
        model.fit(X_train_s, y_train)
        ap = average_precision_score(y_val, model.predict_proba(X_val_s)[:, 1])
        if ap > best_ap:
            best_ap, best_model, best_c = ap, model, C
    return best_model, best_c, best_ap


def save_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    feature_cols: list[str],
    metrics: dict,
    models_dir: str = "models",
) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(models_dir) / timestamp
    save_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, save_path / "model.joblib")
    joblib.dump(scaler, save_path / "scaler.joblib")

    metadata = {
        "timestamp": timestamp,
        "model_params": model.get_params(),
        "feature_cols": feature_cols,
        "metrics": metrics,
    }
    with open(save_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return save_path
