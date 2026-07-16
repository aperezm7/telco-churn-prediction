from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os
import re
import tensorflow as tf

# Suprimir logs molestos de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# APP INIT
app = FastAPI(title="Customer Churn Multi-Model API", version="2.0")

# PATH SETUP
# Subimos un nivel en el directorio para que pueda encontrar la carpeta 'models' desde 'src'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LOAD RESOURCES
print("Cargando Modelos y Preprocesadores en Memoria...")
try:
    # XGBoost
    xgb_model = joblib.load(os.path.join(BASE_DIR, "models", "xgboost", "xgboost_churn_model.pkl"))
    xgb_scaler = joblib.load(os.path.join(BASE_DIR, "models", "xgboost", "scaler_xgb.pkl"))
    xgb_columns = joblib.load(os.path.join(BASE_DIR, "models", "xgboost", "model_columns_xgb.pkl"))
    print("✅ XGBoost cargado")
    
    # Red Neuronal (Keras)
    nn_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "models", "neural_network", "churn_model.h5"))
    nn_scaler = joblib.load(os.path.join(BASE_DIR, "models", "neural_network", "scaler.pkl"))
    nn_columns = joblib.load(os.path.join(BASE_DIR, "models", "neural_network", "model_columns.pkl"))
    print("✅ Red Neuronal cargada")
    
except Exception as e:
    print(f"❌ Error al cargar los archivos: {e}")


# INPUT SCHEMA
class Customer(BaseModel):
    model_type: str = Field(default="xgboost", description="Elige entre 'xgboost' o 'neural_network'")
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 1
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 70.70
    TotalCharges: float = 70.70


# HEALTH CHECK
@app.get("/")
def home():
    return {"status": "Multi-Model API running successfully"}


# PREDICTION ENDPOINT
@app.post("/predict")
def predict(data: Customer):
    try:
        # Convert input -> DataFrame
        data_dict = data.model_dump() if hasattr(data, 'model_dump') else data.dict()
        
        # Validar tipo de modelo y retirarlo de los datos del cliente
        model_choice = data_dict.pop('model_type', 'xgboost').lower()
        if model_choice not in ['xgboost', 'neural_network']:
            raise HTTPException(status_code=400, detail="model_type debe ser 'xgboost' o 'neural_network'")

        df = pd.DataFrame([data_dict])
        
        # One-Hot Encoding
        df_encoded = pd.get_dummies(df)
        
        # Determinar variables a usar según el modelo elegido
        if model_choice == 'xgboost':
            expected_columns = xgb_columns
            scaler = xgb_scaler
        else:
            expected_columns = nn_columns
            scaler = nn_scaler
        
        # Alinear columnas
        df_final = pd.DataFrame(columns=expected_columns)
        for col in expected_columns:
            if col in df_encoded.columns:
                df_final[col] = df_encoded[col]
            else:
                df_final[col] = 0
                
        # Escalar numéricos
        if hasattr(scaler, 'feature_names_in_'):
            num_cols = list(scaler.feature_names_in_)
        else:
            num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
            
        df_final[num_cols] = scaler.transform(df_final[num_cols])
        
        # Predicción Final
        if model_choice == 'xgboost':
            # Ajuste de caracteres (solo XGBoost por seguridad en JSON)
            regex = re.compile(r"\[|\]|<", re.IGNORECASE)
            df_final.columns = [regex.sub("_", col) if any(x in str(col) for x in set('[ ] <')) else col for col in df_final.columns.values]
            
            prediction_prob = float(xgb_model.predict_proba(df_final)[0][1])
            clase = int(prediction_prob > 0.5)
            
        else:
            # Predicción con Keras (TensorFlow)
            X_predict = np.asarray(df_final).astype(np.float32)
            prediction_prob = float(nn_model.predict(X_predict, verbose=0)[0][0])
            clase = int(prediction_prob > 0.5)

        return {
            "churn": clase,
            "probability": prediction_prob,
            "used_model": model_choice,
            "status": "success"
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        return {
            "churn": None,
            "probability": None,
            "status": "error",
            "message": str(e)
        }