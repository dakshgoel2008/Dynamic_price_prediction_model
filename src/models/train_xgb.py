import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import wandb
import argparse
import logging
import joblib
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(data_path, use_wandb=False):
    logging.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Features and Target
    target = 'price'
    # Exclude non-predictive or datetime columns
    features = [
        'distance', 'cab_type', 'name', 'surge_multiplier', 
        'temp', 'clouds', 'pressure', 'rain', 'humidity', 'wind', 
        'day_of_week', 'hour_sin', 'hour_cos'
    ]

    df = df.dropna(subset=features + [target])

    X = df[features].copy()
    y = df[target].copy()

    # Encode categorical variables
    logging.info("Encoding categorical features...")
    cat_cols = ['cab_type', 'name']
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Save encoders for inference later
    os.makedirs('src/models/saved', exist_ok=True)
    joblib.dump(label_encoders, 'src/models/saved/label_encoders.pkl')

    # Train-test split (Note: In a true time-series problem, we'd use TimeSeriesSplit)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        'objective': 'reg:squarederror',
        'n_estimators': 150,
        'learning_rate': 0.1,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }

    if use_wandb:
        wandb.init(project="dynamic-pricing", config=params)
        config = wandb.config
    else:
        config = params

    logging.info("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(**config)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=10
    )

    logging.info("Evaluating model...")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logging.info(f"MAE:  {mae:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"R2:   {r2:.4f}")

    if use_wandb:
        wandb.log({"MAE": mae, "RMSE": rmse, "R2": r2})
        wandb.finish()

    # Save model
    joblib.dump(model, 'src/models/saved/xgboost_baseline.pkl')
    logging.info("Model saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='src/data/processed/merged_data.csv')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for tracking')
    args = parser.parse_args()

    train_model(args.data, use_wandb=args.wandb)
