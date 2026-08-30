import streamlit as st
import joblib
import pandas as pd
import os
import sys

# Ensure src is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.optimization.optimizer import PricingOptimizer

# Page Config
st.set_page_config(
    page_title="Dynamic Pricing Optimizer",
    page_icon="🚕",
    layout="wide"
)

# Load Models
@st.cache_resource
def load_models():
    model_path = 'src/models/saved/xgboost_baseline.pkl'
    encoder_path = 'src/models/saved/label_encoders.pkl'
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        model = joblib.load(model_path)
        label_encoders = joblib.load(encoder_path)
        optimizer = PricingOptimizer(model, label_encoders)
        return optimizer, label_encoders
    return None, None

optimizer, label_encoders = load_models()

# Header
st.title("🚕 Dynamic Pricing & Prescriptive Optimization")
st.markdown("""
This dashboard demonstrates the power of combining predictive Machine Learning (XGBoost) with 
Prescriptive Analytics (Operations Research). 

Adjust the ride parameters on the left, and the optimizer will calculate the exact surge multiplier 
needed to **maximize expected revenue**, accounting for demand elasticity.
""")

if not optimizer:
    st.error("🚨 Model files not found! Please train the model first by running `python src/models/train_xgb.py`")
    st.stop()

# Sidebar for Inputs
st.sidebar.header("Ride Parameters")

col1, col2 = st.sidebar.columns(2)
with col1:
    cab_type = st.selectbox("Cab Provider", ["Uber", "Lyft"])
with col2:
    if cab_type == "Uber":
        name = st.selectbox("Service", ["UberX", "UberXL", "Black", "Black SUV", "WAV", "UberPool"])
    else:
        name = st.selectbox("Service", ["Lyft", "Lyft XL", "Lux", "Lux Black", "Lux Black XL", "Shared"])

distance = st.sidebar.slider("Distance (miles)", 0.1, 10.0, 2.5, 0.1)

st.sidebar.subheader("Weather Conditions")
temp = st.sidebar.slider("Temperature (°F)", 10.0, 100.0, 65.0, 1.0)
rain = st.sidebar.slider("Rain (inches)", 0.0, 1.0, 0.0, 0.05)
humidity = st.sidebar.slider("Humidity", 0.1, 1.0, 0.6, 0.05)
wind = st.sidebar.slider("Wind Speed (mph)", 0.0, 20.0, 5.0, 1.0)
clouds = st.sidebar.slider("Cloud Cover", 0.0, 1.0, 0.5, 0.05)
pressure = st.sidebar.number_input("Pressure (mb)", value=1012.0)

st.sidebar.subheader("Time Features")
day_of_week = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
hour_sin = st.sidebar.slider("Hour (Sine Transform)", -1.0, 1.0, 0.5)
hour_cos = st.sidebar.slider("Hour (Cosine Transform)", -1.0, 1.0, 0.8)

# Encode Inputs safely
try:
    cab_type_encoded = label_encoders['cab_type'].transform([cab_type])[0]
    name_encoded = label_encoders['name'].transform([name])[0]
except ValueError as e:
    st.error(f"Encoding Error: {e}")
    st.stop()

# Create feature dictionary
features = {
    'distance': distance,
    'cab_type': cab_type_encoded,
    'name': name_encoded,
    'temp': temp,
    'clouds': clouds,
    'pressure': pressure,
    'rain': rain,
    'humidity': humidity,
    'wind': wind,
    'day_of_week': day_of_week,
    'hour_sin': hour_sin,
    'hour_cos': hour_cos
}

st.markdown("---")

# Run Optimization
st.subheader("Optimization Results")

with st.spinner('Calculating optimal mathematical surge...'):
    results = optimizer.optimize_surge(features, min_surge=1.0, max_surge=3.0)

# Display Results in large metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Baseline Price (No Surge)", value=f"${results['baseline_price']:.2f}")

with col2:
    st.metric(
        label="Optimal Surge Multiplier", 
        value=f"{results['optimal_surge_multiplier']:.2f}x",
        delta="Revenue Maximized"
    )

with col3:
    st.metric(
        label="Expected Revenue", 
        value=f"${results['expected_revenue']:.2f}",
        delta=f"Win Prob: {int(results['probability_of_acceptance']*100)}%"
    )

st.info(f"**Insight:** To maximize revenue for this {distance} mile {name} trip at {temp}°F, the algorithm prescribes a surge of **{results['optimal_surge_multiplier']:.2f}x**, bringing the final price to **${results['predicted_price_at_optimal_surge']:.2f}**.")
