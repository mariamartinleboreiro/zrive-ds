import json
from pathlib import Path

import joblib
import pandas as pd


DEFAULT_PREDICT_OPTIONS = {
    "models_dir": "p4/models",
    "model_name": None,
    "model_path": None,
}


def build_response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": str(status_code),
        "body": json.dumps(body),
    }


def parse_predict_event(event: dict) -> dict:
    options = {**DEFAULT_PREDICT_OPTIONS}
    for key in options:
        if key in event:
            options[key] = event[key]

    users_raw = event.get("users")
    if users_raw is None:
        raise ValueError("Missing required field: 'users'.")

    if isinstance(users_raw, str):
        users_data = json.loads(users_raw)
    else:
        users_data = users_raw

    if not isinstance(users_data, dict) or not users_data:
        raise ValueError("'users' must be a non-empty dictionary.")

    user_ids = list(users_data.keys())
    X = pd.DataFrame.from_dict(users_data, orient="index")
    return {
        **options,
        "user_ids": user_ids,
        "X": X,
    }


def _resolve_model_path(config: dict) -> Path:
    if config["model_path"]:
        return Path(config["model_path"])

    models_dir = Path(config["models_dir"])
    if config["model_name"]:
        return models_dir / config["model_name"] / "model.joblib"

    model_dirs = sorted(
        [p for p in models_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    if not model_dirs:
        raise ValueError("No trained models found.")

    return model_dirs[0] / "model.joblib"


def handler_predict(event, _):
    config = parse_predict_event(event)
    model_path = _resolve_model_path(config)
    pipeline = joblib.load(model_path)

    proba = pipeline.predict_proba(config["X"])[:, 1]
    prediction = {
        user_id: float(score)
        for user_id, score in zip(config["user_ids"], proba)
    }

    return build_response(
        200,
        {
            "prediction": prediction,
        },
    )
