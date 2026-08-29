import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data(cab_rides_path, weather_path, output_path):
    logging.info("Loading raw datasets...")
    cab_df = pd.read_csv(cab_rides_path)
    weather_df = pd.read_csv(weather_path)

    # 1. Clean Data
    logging.info("Cleaning data...")
    cab_df = cab_df.dropna(subset=['price'])  # Drop target variable NaNs
    weather_df['rain'] = weather_df['rain'].fillna(0) # Fill NaN rain with 0

    # 2. Date Time Processing
    logging.info("Processing timestamps...")
    # Convert timestamps from milliseconds (or seconds) to datetime
    cab_df['datetime'] = pd.to_datetime(cab_df['time_stamp'], unit='ms')
    weather_df['datetime'] = pd.to_datetime(weather_df['time_stamp'], unit='s')
    
    # Sort by datetime for merge_asof
    cab_df = cab_df.sort_values('datetime')
    weather_df = weather_df.sort_values('datetime')

    # 3. Merge Datasets
    logging.info("Merging datasets using nearest timestamp...")
    # We map 'source' in cab rides to 'location' in weather
    merged_df = pd.merge_asof(
        cab_df, weather_df,
        on='datetime',
        left_by='source',
        right_by='location',
        direction='nearest'
    )

    # 4. Feature Engineering
    logging.info("Engineering temporal features...")
    # Extract temporal features
    merged_df['hour'] = merged_df['datetime'].dt.hour
    merged_df['day_of_week'] = merged_df['datetime'].dt.dayofweek

    # Cyclical encoding
    merged_df['hour_sin'] = np.sin(2 * np.pi * merged_df['hour']/24.0)
    merged_df['hour_cos'] = np.cos(2 * np.pi * merged_df['hour']/24.0)
    
    # Drop redundant columns
    cols_to_drop = ['time_stamp_x', 'time_stamp_y', 'location', 'id', 'product_id']
    merged_df = merged_df.drop(columns=[c for c in cols_to_drop if c in merged_df.columns])

    logging.info(f"Final dataset shape: {merged_df.shape}")
    
    # 5. Save Processed Data
    logging.info(f"Saving processed data to {output_path}...")
    merged_df.to_csv(output_path, index=False)
    logging.info("Done!")

if __name__ == "__main__":
    process_data(
        cab_rides_path="src/data/raw/cab_rides.csv",
        weather_path="src/data/raw/weather.csv",
        output_path="src/data/processed/merged_data.csv"
    )
