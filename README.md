# telco-churn-prediction
#### ML model for churn predicition - FastAPI - Dockerfile - compose - Github actions

## Introduction
In the telecommunications sector, customer retention is a key financial driver, as acquiring new users is significantly more expensive than maintaining existing ones. This MLOps  addresses this challenge by deploying an end-to-end classification pipeline trained on historical Telco Customer Churn data. By identifying critical churn indicators and exposing a predictive model through a dockerized FastAPI service, we enable the business to proactively launch targeted customer retention campaigns before users decide to leave.

## Project Architecture & Directory Structure
```text
telco-churn-prediction/
├── .github/workflows/    # CI (GitHub Actions) configurations
├── models/
│   ├── neural_network/   # Trained classification model (joblib)
│   └── xgboost/          # Preprocessing/scaling pipeline (joblib)
├── src/
|   ├──__pycache__/
│   ├── api.py            # FastAPI main application
│   └── __init__.py
├── Dockerfile            # Container deployment blueprint
├── docker-compose.yml    # Local multi-container orchestration
├── requirements.txt      # Python dependencies
└── README.md
