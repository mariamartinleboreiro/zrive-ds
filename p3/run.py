import argparse

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from p3.src.data_loading import load_dataset
from p3.src.preprocessing import build_features, filter_orders
from p3.src.training import save_model, temporal_split, train_and_select


def run(data_path: str, models_dir: str) -> None:
    print("Loading data...")
    df = load_dataset(data_path)

    print("Preprocessing...")
    df = filter_orders(df)
    train_mask, val_mask, test_mask = temporal_split(df)
    X, y = build_features(df)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print("Training...")
    model, best_c, val_ap = train_and_select(X_train_s, y_train, X_val_s, y_val)
    test_ap = average_precision_score(y_test, model.predict_proba(X_test_s)[:, 1])

    metrics = {
        "val_pr_auc": round(val_ap, 4),
        "test_pr_auc": round(test_ap, 4),
        "best_C": best_c,
    }
    print(f"Best C={best_c}  val PR-AUC={val_ap:.4f}  test PR-AUC={test_ap:.4f}")

    path = save_model(model, scaler, X.columns.tolist(), metrics, models_dir=models_dir)
    print(f"Model saved to: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--models-dir", default="p3/models")
    args = parser.parse_args()
    run(args.data_path, args.models_dir)
