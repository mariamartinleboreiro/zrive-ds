import datetime
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


TARGET = "outcome"

NUMERIC_FEATURES = [
    "ordered_before",
    "abandoned_before",
    "active_snoozed",
    "set_as_regular",
    "normalised_price",
    "discount_pct",
    "global_popularity",
    "count_adults",
    "count_children",
    "count_babies",
    "count_pets",
    "people_ex_baby",
    "days_since_purchase_variant_id",
    "avg_days_to_buy_variant_id",
    "std_days_to_buy_variant_id",
    "days_since_purchase_product_type",
    "avg_days_to_buy_product_type",
    "std_days_to_buy_product_type",
    "user_order_seq",
]

CATEGORICAL_FEATURES = ["product_type"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REQUIRED_COLUMNS = [
    "variant_id",
    "product_type",
    "order_id",
    "user_id",
    "created_at",
    "order_date",
    "user_order_seq",
    "outcome",
    "ordered_before",
    "abandoned_before",
    "active_snoozed",
    "set_as_regular",
    "normalised_price",
    "discount_pct",
    "vendor",
    "global_popularity",
    "count_adults",
    "count_children",
    "count_babies",
    "count_pets",
    "people_ex_baby",
    "days_since_purchase_variant_id",
    "avg_days_to_buy_variant_id",
    "std_days_to_buy_variant_id",
    "days_since_purchase_product_type",
    "avg_days_to_buy_product_type",
    "std_days_to_buy_product_type",
]

DEFAULT_MODEL_PARAMETRISATION = {
    "model_type": "hist_gradient_boosting",
    "max_iter": 120,
    "learning_rate": 0.05,
    "max_depth": 4,
    "use_calibration": True,
    "calibration_method": "sigmoid",
}

DEFAULT_FIT_OPTIONS = {
    "data_path": "../data/module_2/box_builder_dataset/feature_frame.csv",
    "models_dir": "p4/models",
    "model_name": None,
    "min_items": 5,
    "train_frac": 0.6,
    "val_frac": 0.2,
}


def build_response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": str(status_code),
        "body": json.dumps(body),
    }


def parse_fit_event(event: dict) -> dict:
    model_parametrisation = event.get("model_parametrisation", {})
    if not isinstance(model_parametrisation, dict):
        raise ValueError("'model_parametrisation' must be a dictionary.")

    merged_model_params = {
        **DEFAULT_MODEL_PARAMETRISATION,
        **model_parametrisation,
    }

    options = {**DEFAULT_FIT_OPTIONS}
    for key in options:
        if key in event:
            options[key] = event[key]

    return {
        "model_parametrisation": merged_model_params,
        **options,
    }


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["order_date", "created_at"])
    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset is empty.")

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    unexpected = set(df[TARGET].dropna().unique()) - {0.0, 1.0}
    if unexpected:
        raise ValueError(f"Unexpected values in 'outcome': {sorted(unexpected)}")


def filter_orders(df: pd.DataFrame, min_items: int = 5) -> pd.DataFrame:
    items_bought = df[df[TARGET] == 1.0].groupby("order_id")["variant_id"].count()
    valid_orders = items_bought[items_bought >= min_items].index
    return df[df["order_id"].isin(valid_orders)].reset_index(drop=True)


def temporal_split(
    df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[pd.Series, pd.Series, pd.Series]:
    orders_by_date = (
        df[["order_id", "order_date"]]
        .drop_duplicates()
        .sort_values("order_date")
        .reset_index(drop=True)
    )
    n_orders = len(orders_by_date)
    train_cutoff = orders_by_date.iloc[int(n_orders * train_frac)]["order_date"]
    test_cutoff = orders_by_date.iloc[int(n_orders * (train_frac + val_frac))]["order_date"]

    train_mask = df["order_date"] < train_cutoff
    val_mask = (df["order_date"] >= train_cutoff) & (df["order_date"] < test_cutoff)
    test_mask = df["order_date"] >= test_cutoff
    return train_mask, val_mask, test_mask


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLUMNS].copy(), df[TARGET].astype(int)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _build_model(model_parametrisation: dict) -> HistGradientBoostingClassifier:
    if model_parametrisation["model_type"] != "hist_gradient_boosting":
        raise ValueError("Only 'hist_gradient_boosting' is supported in this MVP.")

    return HistGradientBoostingClassifier(
        max_iter=model_parametrisation["max_iter"],
        learning_rate=model_parametrisation["learning_rate"],
        max_depth=model_parametrisation["max_depth"],
        random_state=42,
    )


def _resolve_model_name(requested_name: str | None) -> str:
    if requested_name:
        return requested_name
    return f"push_{datetime.datetime.now().strftime('%Y_%m_%d')}"


def _train_pipeline(config: dict) -> tuple[Pipeline, dict]:
    df = load_dataset(config["data_path"])
    df = filter_orders(df, min_items=config["min_items"])

    train_mask, val_mask, _ = temporal_split(
        df,
        train_frac=config["train_frac"],
        val_frac=config["val_frac"],
    )
    X, y = build_xy(df)

    train_val_mask = train_mask | val_mask
    X_train_val = X[train_val_mask]
    y_train_val = y[train_val_mask]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", _build_model(config["model_parametrisation"])),
        ]
    )
    pipeline.fit(X_train_val, y_train_val)

    metadata = {
        "rows_total": int(len(df)),
        "rows_train_val": int(len(X_train_val)),
        "model_parametrisation": config["model_parametrisation"],
        "feature_columns": FEATURE_COLUMNS,
        "min_items": config["min_items"],
        "train_frac": config["train_frac"],
        "val_frac": config["val_frac"],
    }
    return pipeline, metadata


def handler_fit(event, _):
    config = parse_fit_event(event)
    model_name = _resolve_model_name(config["model_name"])

    pipeline, metadata = _train_pipeline(config)

    models_dir = Path(config["models_dir"])
    model_dir = models_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.joblib"

    joblib.dump(pipeline, model_path)
    joblib.dump(metadata, metadata_path)

    return build_response(
        200,
        {
            "model_path": str(model_path),
            "model_name": model_name,
        },
    )
