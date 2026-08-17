import os
import glob
import json
import numpy as np
import pandas as pd

def calculate_aqi_bucket(pm25):
    """
    Calculates a simplified AQI bucket based on PM2.5 concentration.
    Indian AQI PM2.5 breakpoints:
    0-30: Good (1)
    31-60: Satisfactory (2)
    61-90: Moderately polluted (3)
    91-120: Poor (4)
    121-250: Very Poor (5)
    >250: Severe (6)
    """
    if pd.isna(pm25):
        return np.nan
    if pm25 <= 30:
        return 1
    elif pm25 <= 60:
        return 2
    elif pm25 <= 90:
        return 3
    elif pm25 <= 120:
        return 4
    elif pm25 <= 250:
        return 5
    else:
        return 6

def process_station_data(file_path, window_size=12, gap_threshold=6):
    """
    Loads, resamples, imputes, and creates features for a single station.
    """
    df = pd.read_csv(file_path)
    
    # Ensure timestamp is datetime and sort
    if 'timestamp' not in df.columns:
        print(f"Error: 'timestamp' column not found in {file_path}")
        return None

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df = df.set_index('timestamp')
    
    # Resample to hourly, taking the mean for duplicates within an hour
    # For missing hours, this introduces NaNs
    df = df.resample('h').mean()
    
    # Columns expected (some might be missing, handled dynamically)
    features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'temperature', 'humidity']
    available_features = [col for col in features if col in df.columns]
    
    if 'PM2.5' not in available_features:
        print(f"Warning: PM2.5 not found in {file_path}. Skipping this station as target cannot be created.")
        return None

    # Handle missing values: Interpolate short gaps, flag large gaps
    for col in available_features:
        is_nan = df[col].isna()
        
        # Group consecutive NaNs
        blocks = (~is_nan).cumsum()
        block_sizes = is_nan.groupby(blocks).sum()
        
        # Identify large gaps
        large_gap_blocks = block_sizes[block_sizes > gap_threshold].index
        large_gap_mask = is_nan & blocks.isin(large_gap_blocks)
        
        # Interpolate
        interpolated = df[col].interpolate(method='linear')
        
        # Re-apply NaNs to large gaps (they remain missing and will be dropped later)
        interpolated[large_gap_mask] = np.nan
        df[col] = interpolated

    # Target: next-hour PM2.5 and AQI bucket
    df['target_pm25'] = df['PM2.5'].shift(-1)
    df['target_aqi'] = df['target_pm25'].apply(calculate_aqi_bucket)
    
    # Create window features
    window_cols = []
    for w in range(window_size):
        for col in available_features:
            shifted = df[col].shift(w)
            window_cols.append(shifted.rename(f"{col}_lag_{w}"))
            
    # Combine everything
    df_windowed = pd.concat(window_cols + [df[['target_pm25', 'target_aqi']]], axis=1)
    
    # Drop rows with any NaN in features or target
    # This automatically drops samples overlapping with large gaps, as well as the first window_size-1 and the last row
    df_windowed = df_windowed.dropna()
    
    return df_windowed

def split_data(df, train_frac=0.6, val_frac=0.2):
    """
    Splits data chronologically.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df

def create_clients_data(data_dir, output_dir, config, window_size=12, gap_threshold=6):
    """
    Orchestrates data loading, preprocessing, and saving for multiple stations.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    station_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    print(f"Found {len(station_files)} station CSV files in '{data_dir}'.")
    
    station_metadata = []
    
    # config mapping station_id to metadata (e.g., city, regime)
    station_config_map = config if config else {}

    for file_path in station_files:
        station_id = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\nProcessing station: {station_id}")
        
        df_processed = process_station_data(file_path, window_size=window_size, gap_threshold=gap_threshold)
        
        if df_processed is None or df_processed.empty:
            print(f"  -> Skipping {station_id} due to insufficient valid data.")
            continue
            
        print(f"  -> Time range: {df_processed.index.min()} to {df_processed.index.max()}")
        print(f"  -> Total valid samples (after dropping NaNs): {len(df_processed)}")
        
        # Split data
        train_df, val_df, test_df = split_data(df_processed)
        print(f"  -> Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Save splits
        station_out_dir = os.path.join(output_dir, station_id)
        os.makedirs(station_out_dir, exist_ok=True)
        
        train_df.to_parquet(os.path.join(station_out_dir, "train.parquet"))
        val_df.to_parquet(os.path.join(station_out_dir, "val.parquet"))
        test_df.to_parquet(os.path.join(station_out_dir, "test.parquet"))
        
        # Collect metadata
        meta = station_config_map.get(station_id, {
            "city": "Unknown",
            "regime": "Unknown"
        })
        meta["station_id"] = station_id
        meta["num_samples"] = len(df_processed)
        station_metadata.append(meta)

    # Save small JSON config
    config_path = os.path.join(output_dir, "stations_config.json")
    with open(config_path, 'w') as f:
        json.dump(station_metadata, f, indent=4)
        
    print(f"\nSaved station metadata to {config_path}")
    print("Data processing complete.")

if __name__ == "__main__":
    # Example local test
    sample_data_dir = "sample_data"
    sample_out_dir = "processed_data"
    
    if not os.path.exists(sample_data_dir) or len(glob.glob(os.path.join(sample_data_dir, "*.csv"))) < 3:
        os.makedirs(sample_data_dir, exist_ok=True)
        print(f"Creating sample dummy data in {sample_data_dir}/ for testing...")
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='h')
        for i in range(1, 4):
            df = pd.DataFrame({
                'timestamp': dates, 
                'PM2.5': np.random.uniform(10, 200, len(dates)), 
                'temperature': np.random.uniform(15, 35, len(dates))
            })
            # Introduce a small gap (interpolate)
            df.loc[10:12, 'PM2.5'] = np.nan 
            # Introduce a large gap (drop)
            df.loc[100:110, 'PM2.5'] = np.nan 
            df.to_csv(os.path.join(sample_data_dir, f"station_{i}.csv"), index=False)
        
    sample_config = {
        f"station_{i}": {"city": "Delhi", "regime": "traffic"} for i in range(1, 4)
    }
    
    print("Running data_loader.py...")
    create_clients_data(sample_data_dir, sample_out_dir, sample_config)
