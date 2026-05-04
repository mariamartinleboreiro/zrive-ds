import logging
import os
import sys
import time
from contextlib import asynccontextmanager

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from src.basket_model.basket_model import BasketModel
from src.basket_model.feature_store import FeatureStore
from src.exceptions import PredictionException, UserNotFoundException

LOG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/metrics.txt"))
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.feature_store = FeatureStore()
    app.state.model = BasketModel()
    logger.info("Service started — feature store and model loaded")
    yield
    logger.info("Service shutting down")


app = FastAPI(lifespan=lifespan)


class PredictRequest(BaseModel):
    user_id: str


class PredictResponse(BaseModel):
    user_id: str
    predicted_basket_value: float


@app.get("/status")
def status():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, raw_request: Request):
    start = time.time()
    user_id = request.user_id
    try:
        features = raw_request.app.state.feature_store.get_features(user_id)
        feature_row = features.to_frame().T if isinstance(features, pd.Series) else features.iloc[[-1]]
        prediction = raw_request.app.state.model.predict(feature_row)
        predicted_value = float(prediction[0])
        latency_ms = (time.time() - start) * 1000
        logger.info("PREDICT user_id=%s predicted_basket_value=%.4f latency_ms=%.2f", user_id, predicted_value, latency_ms)
        return PredictResponse(user_id=user_id, predicted_basket_value=predicted_value)

    except UserNotFoundException as exc:
        latency_ms = (time.time() - start) * 1000
        logger.warning("USER_NOT_FOUND user_id=%s latency_ms=%.2f", user_id, latency_ms)
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except PredictionException as exc:
        latency_ms = (time.time() - start) * 1000
        logger.error("PREDICTION_ERROR user_id=%s latency_ms=%.2f error=%s", user_id, latency_ms, str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
