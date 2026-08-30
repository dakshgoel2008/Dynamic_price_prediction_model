import pandas as pd
import numpy as np
import logging
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from econml.dml import LinearDML
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def estimate_causal_effect(data_path, output_dir):
    logging.info("Loading data for Causal Inference...")
    df = pd.read_csv(data_path)

    # We want to estimate the effect of Surge Multiplier (T) on Price (Y)
    # Conditioning on confounders (X and W)
    
    features = [
        'distance', 'cab_type', 'surge_multiplier', 'price',
        'temp', 'rain', 'hour_sin', 'hour_cos'
    ]
    df = df.dropna(subset=features)
    
    # Subsample for faster causal estimation (EconML can be heavy)
    df = df.sample(n=50000, random_state=42)

    le = LabelEncoder()
    df['cab_type_encoded'] = le.fit_transform(df['cab_type'])

    Y = df['price'].values
    T = df['surge_multiplier'].values
    
    # X: Features that cause heterogeneity in the treatment effect
    X = df[['distance', 'cab_type_encoded']].values 
    
    # W: Confounders that affect both Treatment and Outcome but we don't care about their heterogeneity
    W = df[['temp', 'rain', 'hour_sin', 'hour_cos']].values

    logging.info("Initializing Double Machine Learning (DML) estimator...")
    # DML uses ML models to residualize T and Y, then estimates the causal effect
    est = LinearDML(
        model_y=XGBRegressor(max_depth=3, n_estimators=100),
        model_t=XGBRegressor(max_depth=3, n_estimators=100),
        discrete_treatment=False,
        linear_first_stages=False,
        cv=3
    )

    logging.info("Fitting Causal Model (this may take a minute)...")
    est.fit(Y, T, X=X, W=W)

    # Get the Constant Marginal Treatment Effect (ATE)
    ate = est.ate(X)
    logging.info(f"Average Treatment Effect (ATE) of Surge Multiplier on Price: ${ate:.2f}")

    # Plot the relationship directly
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['distance'], y=df['price'], hue=df['surge_multiplier'], alpha=0.6, palette="viridis")
    plt.title("Impact of Distance and Surge Multiplier on Price")
    plt.xlabel("Trip Distance (miles)")
    plt.ylabel("Ride Price ($)")
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'causal_effect_distance.png')
    plt.savefig(plot_path)
    logging.info(f"Saved Causal Effect plot to {plot_path}")
    
    # Provide a business summary
    summary = (
        f"\n--- CAUSAL INFERENCE SUMMARY ---\n"
        f"Average Treatment Effect (ATE): +${ate:.2f}\n"
        f"Interpretation: On average, increasing the surge multiplier by 1 unit causes the price to increase by ${ate:.2f}, "
        f"holding weather, time, and distance constant.\n"
    )
    logging.info(summary)
    
    with open(os.path.join(output_dir, 'causal_summary.txt'), 'w') as f:
        f.write(summary)


if __name__ == "__main__":
    estimate_causal_effect(
        data_path='src/data/processed/merged_data.csv',
        output_dir='src/causal/results'
    )
