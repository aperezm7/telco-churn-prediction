import requests

def test_predict():
    json_request = {
      "model_type": "xgboost",
      "gender": "Female",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 1,
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "No",
      "OnlineBackup": "No",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "No",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 70.7,
      "TotalCharges": 70.7
    }

    response = requests.post(
        "http://localhost:8000/predict", json=json_request
    )
    assert response.status_code == 200
    data = response.json()
    assert data["churn"] == 1
    assert data["probability"] >= 0.9
    assert data["used_model"] == "xgboost"
