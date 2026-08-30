import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import pandas as pd
import logging

from src.optimization.optimizer import PricingOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Dynamic Pricing & Optimization API",
    description="Predict ride prices and optimize surge multipliers for maximum revenue.",
    version="1.0.0"
)

@app.get("/", include_in_schema=False)
def read_root():
    return RedirectResponse(url="/docs")

# Global variables for models
model = None
label_encoders = None
optimizer = None

class RideRequest(BaseModel):
    distance: float
    cab_type: str
    name: str
    surge_multiplier: float = 1.0
    temp: float
    clouds: float
    pressure: float
    rain: float
    humidity: float
    wind: float
    day_of_week: int
    hour_sin: float
    hour_cos: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "distance": 2.5,
                "cab_type": "Uber",
                "name": "UberX",
                "surge_multiplier": 1.0,
                "temp": 65.0,
                "clouds": 0.5,
                "pressure": 1012.0,
                "rain": 0.0,
                "humidity": 0.6,
                "wind": 5.0,
                "day_of_week": 2,
                "hour_sin": 0.5,
                "hour_cos": 0.8
            }
        }
    }

@app.on_event("startup")
def load_models():
    global model, label_encoders, optimizer
    
    model_path = 'src/models/saved/xgboost_baseline.pkl'
    encoder_path = 'src/models/saved/label_encoders.pkl'
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        logging.info("Loading XGBoost model and Label Encoders...")
        model = joblib.load(model_path)
        label_encoders = joblib.load(encoder_path)
        optimizer = PricingOptimizer(model, label_encoders)
        logging.info("Models loaded successfully.")
    else:
        logging.warning("Model files not found. Please train the model first.")

def preprocess_input(request: RideRequest):
    """Convert API request to a dictionary with encoded categorical features."""
    if label_encoders is None:
        raise HTTPException(status_code=500, detail="Models are not loaded.")
        
    data = request.dict()
    
    # Encode categorical variables
    try:
        data['cab_type'] = label_encoders['cab_type'].transform([data['cab_type']])[0]
        data['name'] = label_encoders['name'].transform([data['name']])[0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Categorical encoding error: {str(e)}. Check cab_type and name values.")
        
    return data

@app.post("/predict")
def predict_price(request: RideRequest):
    """Predict the ride price for a given set of features including surge_multiplier."""
    features = preprocess_input(request)
    
    feature_cols = [
        'distance', 'cab_type', 'name', 'surge_multiplier', 
        'temp', 'clouds', 'pressure', 'rain', 'humidity', 'wind', 
        'day_of_week', 'hour_sin', 'hour_cos'
    ]
    df_pred = pd.DataFrame([features])[feature_cols]
    
    price = model.predict(df_pred)[0]
    
    return {
        "predicted_price": round(float(price), 2),
        "surge_multiplier": request.surge_multiplier
    }

@app.post("/optimize")
def optimize_surge(request: RideRequest):
    """Find the optimal surge multiplier to maximize expected revenue."""
    if optimizer is None:
        raise HTTPException(status_code=500, detail="Optimizer is not initialized.")
        
    features = preprocess_input(request)
    # The optimizer will vary the surge_multiplier itself, so we don't need to pass it explicitly
    
    result = optimizer.optimize_surge(features, min_surge=1.0, max_surge=3.0)
    return result

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": model is not None}
