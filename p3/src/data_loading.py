import pandas as pd


REQUIRED_COLUMNS = [
    "variant_id", "product_type", "order_id", "user_id",
    "created_at", "order_date", "user_order_seq", "outcome",
    "ordered_before", "abandoned_before", "active_snoozed", "set_as_regular",
    "normalised_price", "discount_pct", "vendor", "global_popularity",
    "count_adults", "count_children", "count_babies", "count_pets", "people_ex_baby",
    "days_since_purchase_variant_id", "avg_days_to_buy_variant_id", "std_days_to_buy_variant_id",
    "days_since_purchase_product_type", "avg_days_to_buy_product_type", "std_days_to_buy_product_type",
]


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["order_date", "created_at"])
    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset is empty.")

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    unexpected = set(df["outcome"].dropna().unique()) - {0.0, 1.0}
    if unexpected:
        raise ValueError(f"Unexpected values in 'outcome': {unexpected}")
