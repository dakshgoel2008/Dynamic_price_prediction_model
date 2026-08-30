import numpy as np
from scipy.optimize import minimize_scalar
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PricingOptimizer:
    def __init__(self, model, label_encoders):
        """
        Initialize the optimizer with a trained predictive model and label encoders.
        """
        self.model = model
        self.label_encoders = label_encoders
        # Demand elasticity: For every $1 increase above baseline (surge 1.0), 
        # probability of acceptance drops by 5%
        self.elasticity_factor = 0.05 

    def _predict_price(self, features, surge_multiplier):
        """Helper to predict price for a specific surge multiplier."""
        features_copy = features.copy()
        features_copy['surge_multiplier'] = surge_multiplier
        
        # XGBoost expects a DataFrame with the same columns as during training
        feature_cols = [
            'distance', 'cab_type', 'name', 'surge_multiplier', 
            'temp', 'clouds', 'pressure', 'rain', 'humidity', 'wind', 
            'day_of_week', 'hour_sin', 'hour_cos'
        ]
        
        # Ensure we just pass a single row DF
        df_pred = pd.DataFrame([features_copy])
        df_pred = df_pred[feature_cols]
        
        price = self.model.predict(df_pred)[0]
        return price

    def expected_revenue(self, surge_multiplier, features, baseline_price):
        """
        Calculate expected revenue based on predicted price and heuristic demand.
        Revenue = Price * P(Acceptance)
        P(Acceptance) = max(0, 1 - elasticity_factor * (Price - Baseline_Price))
        """
        predicted_price = self._predict_price(features, surge_multiplier)
        
        # Calculate demand dropoff
        price_diff = max(0, predicted_price - baseline_price)
        p_acceptance = max(0.0, 1.0 - (self.elasticity_factor * price_diff))
        
        return predicted_price * p_acceptance

    def optimize_surge(self, features_dict, min_surge=1.0, max_surge=3.0):
        """
        Finds the optimal surge multiplier that maximizes expected revenue.
        features_dict: dictionary containing all required features EXCEPT surge_multiplier.
        """
        # Calculate baseline price (Surge = 1.0)
        baseline_price = float(self._predict_price(features_dict, surge_multiplier=1.0))
        logging.info(f"Baseline Price (Surge=1.0) estimated at: ${baseline_price:.2f}")

        # We want to maximize expected_revenue, which means minimizing its negative
        def objective_function(surge):
            return -self.expected_revenue(surge, features_dict, baseline_price)

        # Use scipy to find the minimum of the negative revenue function
        # Bounded scalar optimization since we only optimize the single scalar 'surge_multiplier'
        result = minimize_scalar(
            objective_function, 
            bounds=(min_surge, max_surge), 
            method='bounded'
        )

        optimal_surge = float(result.x)
        max_revenue = -float(result.fun)
        optimal_price = float(self._predict_price(features_dict, optimal_surge))
        
        return {
            "baseline_price": round(baseline_price, 2),
            "optimal_surge_multiplier": round(optimal_surge, 2),
            "predicted_price_at_optimal_surge": round(optimal_price, 2),
            "expected_revenue": round(max_revenue, 2),
            "probability_of_acceptance": round(max_revenue / optimal_price, 2) if optimal_price > 0 else 0
        }
