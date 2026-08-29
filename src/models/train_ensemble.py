import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import logging
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_ensemble(data_path):
    logging.info("Loading data for Ensembling...")
    df = pd.read_csv(data_path)
    
    features = [
        'distance', 'cab_type', 'name', 'surge_multiplier', 
        'temp', 'clouds', 'pressure', 'rain', 'humidity', 'wind', 
        'day_of_week', 'hour_sin', 'hour_cos'
    ]
    df = df.dropna(subset=features + ['price'])

    X = df[features].copy()
    y = df['price'].copy()

    # Encode categorical variables
    le_cab = LabelEncoder()
    le_name = LabelEncoder()
    X['cab_type'] = le_cab.fit_transform(X['cab_type'])
    X['name'] = le_name.fit_transform(X['name'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. XGBoost
    logging.info("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # 2. LightGBM
    logging.info("Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    lgb_model.fit(X_train, y_train)

    # Ensembling (Simple Averaging)
    logging.info("Evaluating Ensemble...")
    xgb_preds = xgb_model.predict(X_test)
    lgb_preds = lgb_model.predict(X_test)
    
    # Blended predictions
    ensemble_preds = (xgb_preds + lgb_preds) / 2.0
    
    mae = mean_absolute_error(y_test, ensemble_preds)
    r2 = r2_score(y_test, ensemble_preds)
    
    logging.info(f"--- ENSEMBLE RESULTS ---")
    logging.info(f"Ensemble MAE: ${mae:.2f}")
    logging.info(f"Ensemble R2:  {r2:.4f}")
    
    # Save the models
    joblib.dump({'xgb': xgb_model, 'lgb': lgb_model}, 'src/models/saved/ensemble_models.pkl')
    logging.info("Ensemble models saved successfully.")

if __name__ == "__main__":
    train_ensemble('src/data/processed/merged_data.csv')
