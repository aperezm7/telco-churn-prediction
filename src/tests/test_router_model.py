import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


def test_predict_xgboost_returns_success():
    # Fake column schema the mocked model expects
    columns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
               "gender_Female", "gender_Male"]

    # Fake scaler: passes numbers through unchanged
    scaler = MagicMock()
    scaler.feature_names_in_ = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    scaler.transform.side_effect = lambda X: np.asarray(X)

    # Fake XGBoost model: always predicts 73% churn probability
    xgb_model = MagicMock()
    xgb_model.predict_proba.return_value = np.array([[0.27, 0.73]])

    def fake_joblib_load(path, *a, **k):
        if "xgboost_churn_model" in path:
            return xgb_model
        if "scaler_xgb" in path:
            return scaler
        if "model_columns_xgb" in path:
            return list(columns)
        return MagicMock()  # neural network files, not used in this test

    with patch("joblib.load", side_effect=fake_joblib_load), \
         patch("tensorflow.keras.models.load_model", return_value=MagicMock()):
        sys.modules.pop("api", None)
        import api  # loads with the mocks above instead of real files

        client = TestClient(api.app)
        response = client.post("/predict", json={"model_type": "xgboost"})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["used_model"] == "xgboost"
        assert abs(body["probability"] - 0.73) < 1e-6
        assert body["churn"] == 1