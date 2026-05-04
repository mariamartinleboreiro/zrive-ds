from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.exceptions import PredictionException, UserNotFoundException
from src.module_6.app import app


@pytest.fixture
def client():
    with patch("src.module_6.app.FeatureStore") as mock_fs_cls, \
         patch("src.module_6.app.BasketModel") as mock_model_cls:

        mock_fs = MagicMock()
        mock_fs.get_features.return_value = MagicMock(values=np.array([10.0, 3, 2, 5]))
        mock_fs_cls.return_value = mock_fs

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([42.5])
        mock_model_cls.return_value = mock_model

        with TestClient(app) as c:
            yield c


def test_status(client):
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_success(client):
    response = client.post("/predict", json={"user_id": "user_001"})
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "user_001"
    assert data["predicted_basket_value"] == pytest.approx(42.5)


def test_predict_user_not_found(client):
    client.app.state.feature_store.get_features.side_effect = UserNotFoundException("User not found in feature store")
    response = client.post("/predict", json={"user_id": "unknown_user"})
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_predict_model_error(client):
    client.app.state.feature_store.get_features.side_effect = None
    client.app.state.feature_store.get_features.return_value = MagicMock(values=np.array([10.0, 3, 2, 5]))
    client.app.state.model.predict.side_effect = PredictionException("Inference failed")
    response = client.post("/predict", json={"user_id": "user_001"})
    assert response.status_code == 500
    assert "inference failed" in response.json()["detail"].lower()
