import pandas as pd


NUMERIC_FEATURES = [
    "ordered_before", "abandoned_before", "active_snoozed", "set_as_regular",
    "normalised_price", "discount_pct", "global_popularity",
    "count_adults", "count_children", "count_babies", "count_pets", "people_ex_baby",
    "days_since_purchase_variant_id", "avg_days_to_buy_variant_id", "std_days_to_buy_variant_id",
    "days_since_purchase_product_type", "avg_days_to_buy_product_type", "std_days_to_buy_product_type",
    "user_order_seq",
]
TARGET = "outcome"


def filter_orders(df: pd.DataFrame, min_items: int = 5) -> pd.DataFrame:
    items_bought = df[df[TARGET] == 1.0].groupby("order_id")["variant_id"].count()
    valid_orders = items_bought[items_bought >= min_items].index
    return df[df["order_id"].isin(valid_orders)].reset_index(drop=True)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.get_dummies(df, columns=["product_type"], drop_first=True)
    pt_cols = [c for c in df.columns if c.startswith("product_type_")]
    feature_cols = NUMERIC_FEATURES + pt_cols
    return df[feature_cols], df[TARGET]
