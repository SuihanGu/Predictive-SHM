# -*- coding: utf-8 -*-
"""
Crack Meter Prediction Service
Fetch data hourly and make predictions using the model
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pickle
import requests
from datetime import datetime, timedelta
import time
import json
import os
from threading import Timer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ==========================================
# Model Architecture Definition (Identical to Training)
# ==========================================

class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, ff_hidden_dim, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.self_ff = nn.Sequential(
            nn.Linear(input_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, input_dim))
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.self_ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Conv1dLayer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dropout):
        super(Conv1dLayer, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=output_dim, 
                               kernel_size=kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.dropout(self.relu(x))
        x = x.permute(0, 2, 1)
        return x


class TransformerCnn(nn.Module):
    def __init__(self, response_dim, env_dim, trans_dim, num_heads, ff_hidden_dim, 
                 conv_hidden_dim, kernel_size, dropout, n_steps, lag, m):
        super(TransformerCnn, self).__init__()
        self.response_dim = response_dim
        self.n_steps = n_steps
        self.transformer = TransformerEncoderLayer(input_dim=trans_dim, num_heads=num_heads, 
                                                   ff_hidden_dim=ff_hidden_dim, dropout=dropout)
        self.conv_response = Conv1dLayer(input_dim=response_dim, output_dim=conv_hidden_dim, 
                                         kernel_size=kernel_size, dropout=dropout)
        self.conv_env = Conv1dLayer(input_dim=env_dim, output_dim=conv_hidden_dim, 
                                    kernel_size=kernel_size, dropout=dropout)
        self.conv_trans = Conv1dLayer(input_dim=trans_dim, output_dim=conv_hidden_dim, 
                                      kernel_size=kernel_size, dropout=dropout)
        self.final_conv = Conv1dLayer(input_dim=conv_hidden_dim, output_dim=response_dim, 
                                      kernel_size=kernel_size, dropout=dropout)
        fc_input_dim = response_dim * (lag + m * 2)
        self.fc = nn.Linear(fc_input_dim, response_dim * n_steps)
    
    def forward(self, x_response, x_env, x_cat):
        x_cat = self.transformer(x_cat)
        x_response_conv = self.conv_response(x_response)
        x_env_conv = self.conv_env(x_env)
        x_cat_conv = self.conv_trans(x_cat)
        x_concat = torch.cat([x_response_conv, x_env_conv, x_cat_conv], dim=1)
        x_final_conv = self.final_conv(x_concat)
        x_final_flat = x_final_conv.reshape(x_final_conv.size(0), -1)
        x_final_fc = self.fc(x_final_flat)
        output = x_final_fc.view(x_final_conv.size(0), -1, self.response_dim)
        return output


# ==========================================
# Configuration and Global Variables
# ==========================================

# API Base URL
API_BASE_URL = "http://139.159.136.213:4999/iem/shm"

# Device Numbers
CRACK_NUMBERS = ["623622", "623641", "623628"]  # crack_1, crack_2, crack_3
SETTLEMENT_NUMBERS = ["004521", "004548", "004591", "152947"]  # settlement_1~4
TILT_NUMBERS = ["00476464", "00476465", "00476466", "00476467"]  # tilt_x/y_1~4
WATER_LEVEL_NUMBER = "478967"  # water_level

# Model Configuration (Crack Prediction)
MODEL_CONFIG = {
    'response_dim': 3,
    'env_dim': 2,
    'trans_dim': 15,
    'num_heads': 3,
    'ff_hidden_dim': 128,
    'conv_hidden_dim': 96,
    'kernel_size': 3,
    'dropout': 0.25,
    'n_steps': 6,  # Predict 6 future time steps (60 minutes)
    'm': 30,  # Input 30 time steps (300 minutes)
    'lag': 80  # Environmental lag of 80 time steps (800 minutes)
}

# Model and Scaler Paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_crack_model.pth')
SCALER_ALL_PATH = os.path.join(MODEL_DIR, 'scaler_all.pkl')
SCALER_RESPONSE_PATH = os.path.join(MODEL_DIR, 'scaler_response.pkl')

# Global Variables
model = None
scaler_all = None
scaler_response = None
device = None
latest_predictions = None  # Store latest prediction results (for backward compatibility)
# Store all prediction results, key is time string, value is prediction data point
prediction_storage: dict[str, dict] = {}  # {time: {time, timestamp, crack_1, crack_2, crack_3}}


# ==========================================
# Data Fetching Functions
# ==========================================

def get_timestamp_range(hours_back=24):
    """Get timestamp range"""
    now = datetime.now()
    start = now - timedelta(hours=hours_back)
    timestamp1 = int(start.timestamp())
    timestamp2 = int(now.timestamp())
    return timestamp1, timestamp2


def fetch_sensor_data(api_path, timestamp1, timestamp2):
    """Fetch sensor data"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/{api_path}",
            params={"timestamp1": timestamp1, "timestamp2": timestamp2},
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and 'data' in data:
                return data['data']
            elif isinstance(data, list):
                return data
            else:
                return []
        else:
            print(f"API request failed: {api_path}, status code: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching data: {api_path}, error: {e}")
        return []


def fetch_all_sensor_data():
    """Fetch all sensor data"""
    timestamp1, timestamp2 = get_timestamp_range(hours_back=48)  # Fetch 48 hours of data to ensure sufficient history
    
    print(f"Fetching sensor data... (Time range: {datetime.fromtimestamp(timestamp1)} to {datetime.fromtimestamp(timestamp2)})")
    
    # Fetch all data in parallel
    crack_data = fetch_sensor_data("jmData", timestamp1, timestamp2)
    tilt_data = fetch_sensor_data("jmBus", timestamp1, timestamp2)
    level_data = fetch_sensor_data("jmLevel", timestamp1, timestamp2)
    water_level_data = fetch_sensor_data("jmWlg", timestamp1, timestamp2)
    
    print(f"Data fetched: crack={len(crack_data)}, tilt={len(tilt_data)}, settlement={len(level_data)}, water_level={len(water_level_data)}")
    
    return {
        'crack': crack_data,
        'tilt': tilt_data,
        'level': level_data,
        'water_level': water_level_data
    }


def process_sensor_data_to_dataframe(sensor_data):
    """Convert sensor data to DataFrame (aligned to 10-minute intervals)"""
    # Create time series (10-minute intervals)
    now = datetime.now()
    start_time = now - timedelta(hours=48)
    
    # Generate time series with 10-minute intervals
    time_points = []
    # Round start time to 10 minutes
    start_minutes = start_time.minute
    rounded_start_minutes = ((start_minutes // 10) + (1 if start_minutes % 10 >= 5 else 0)) * 10
    if rounded_start_minutes >= 60:
        rounded_start_minutes = 0
        current = start_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        current = start_time.replace(minute=rounded_start_minutes, second=0, microsecond=0)
    
    while current <= now:
        time_points.append(current)
        current += timedelta(minutes=10)
    
    # Initialize data dictionary
    data_dict = {'time': [t.strftime("%Y-%m-%d %H:%M:%S") for t in time_points]}
    
    # Initialize all columns
    for i in range(4):
        data_dict[f'settlement_{i+1}'] = [None] * len(time_points)
    for i in range(3):
        data_dict[f'crack_{i+1}'] = [None] * len(time_points)
    for i in range(4):
        data_dict[f'tilt_x_{i+1}'] = [None] * len(time_points)
        data_dict[f'tilt_y_{i+1}'] = [None] * len(time_points)
    data_dict['water_level'] = [None] * len(time_points)
    data_dict['temperature'] = [None] * len(time_points)
    
    # Process crack meter data
    for item in sensor_data['crack']:
        number = str(item.get('number', ''))
        if number in CRACK_NUMBERS:
            idx = CRACK_NUMBERS.index(number)
            timestamp = item.get('timestamp')
            if timestamp:
                ts = int(timestamp) if isinstance(timestamp, str) else timestamp
                dt = datetime.fromtimestamp(ts)
                # Round to 10 minutes
                minutes = dt.minute
                rounded_minutes = ((minutes // 10) + (1 if minutes % 10 >= 5 else 0)) * 10
                if rounded_minutes >= 60:
                    rounded_minutes = 0
                    dt = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                else:
                    dt = dt.replace(minute=rounded_minutes, second=0, microsecond=0)
                
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                if time_str in data_dict['time']:
                    time_idx = data_dict['time'].index(time_str)
                    value = item.get('data1')
                    if value is not None:
                        try:
                            data_dict[f'crack_{idx+1}'][time_idx] = float(value)
                        except:
                            pass
    
    # Process settlement data
    for item in sensor_data['level']:
        number = str(item.get('number', ''))
        if number in SETTLEMENT_NUMBERS:
            idx = SETTLEMENT_NUMBERS.index(number)
            timestamp = item.get('timestamp')
            if timestamp:
                ts = int(timestamp) if isinstance(timestamp, str) else timestamp
                dt = datetime.fromtimestamp(ts)
                minutes = dt.minute
                rounded_minutes = ((minutes // 10) + (1 if minutes % 10 >= 5 else 0)) * 10
                if rounded_minutes >= 60:
                    rounded_minutes = 0
                    dt = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                else:
                    dt = dt.replace(minute=rounded_minutes, second=0, microsecond=0)
                
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                if time_str in data_dict['time']:
                    time_idx = data_dict['time'].index(time_str)
                    value = item.get('data1')
                    if value is not None:
                        try:
                            data_dict[f'settlement_{idx+1}'][time_idx] = float(value)
                        except:
                            pass
    
    # Process tilt data
    for item in sensor_data['tilt']:
        number = str(item.get('number', ''))
        if number in TILT_NUMBERS:
            idx = TILT_NUMBERS.index(number)
            timestamp = item.get('timestamp')
            if timestamp:
                ts = int(timestamp) if isinstance(timestamp, str) else timestamp
                dt = datetime.fromtimestamp(ts)
                minutes = dt.minute
                rounded_minutes = ((minutes // 10) + (1 if minutes % 10 >= 5 else 0)) * 10
                if rounded_minutes >= 60:
                    rounded_minutes = 0
                    dt = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                else:
                    dt = dt.replace(minute=rounded_minutes, second=0, microsecond=0)
                
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                if time_str in data_dict['time']:
                    time_idx = data_dict['time'].index(time_str)
                    # X direction
                    value_x = item.get('data1')
                    if value_x is not None:
                        try:
                            data_dict[f'tilt_x_{idx+1}'][time_idx] = float(value_x)
                        except:
                            pass
                    # Y direction
                    value_y = item.get('data2')
                    if value_y is not None:
                        try:
                            data_dict[f'tilt_y_{idx+1}'][time_idx] = float(value_y)
                        except:
                            pass
                    # Temperature (use data3 from first tilt sensor)
                    if idx == 0:
                        value_temp = item.get('data3')
                        if value_temp is not None:
                            try:
                                data_dict['temperature'][time_idx] = float(value_temp)
                            except:
                                pass
    
    # Process water level data
    for item in sensor_data['water_level']:
        number = str(item.get('number', ''))
        if number == WATER_LEVEL_NUMBER or number.replace('0', '') == WATER_LEVEL_NUMBER.replace('0', ''):
            timestamp = item.get('timestamp')
            if timestamp:
                ts = int(timestamp) if isinstance(timestamp, str) else timestamp
                dt = datetime.fromtimestamp(ts)
                minutes = dt.minute
                rounded_minutes = ((minutes // 10) + (1 if minutes % 10 >= 5 else 0)) * 10
                if rounded_minutes >= 60:
                    rounded_minutes = 0
                    dt = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                else:
                    dt = dt.replace(minute=rounded_minutes, second=0, microsecond=0)
                
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                if time_str in data_dict['time']:
                    time_idx = data_dict['time'].index(time_str)
                    value = item.get('data1')
                    if value is not None:
                        try:
                            data_dict['water_level'][time_idx] = float(value)
                        except:
                            pass
    
    # Create DataFrame
    df = pd.DataFrame(data_dict)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Arrange columns in order (consistent with training)
    columns_order = (
        [f'settlement_{i+1}' for i in range(4)] +
        [f'crack_{i+1}' for i in range(3)] +
        [f'tilt_x_{i+1}' for i in range(4)] +
        [f'tilt_y_{i+1}' for i in range(4)] +
        ['water_level', 'temperature']
    )
    
    df = df[columns_order]
    
    return df


# ==========================================
# Model Loading and Prediction
# ==========================================
# 步骤 1.1 可插拔模型：可改用 backend/adapters/model_adapter 切换模型，例如：
#   from adapters.model_adapter import get_adapter, MODEL_REGISTRY
#   adapter = get_adapter("transformer_cnn"); out = adapter.predict((x_res, x_env, x_cat))
# 当前仍使用下方 load_model_and_scalers() 以保持兼容。

def load_model_and_scalers():
    """Load model and scalers"""
    global model, scaler_all, scaler_response, device
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
    
    # Create model
    model = TransformerCnn(
        response_dim=MODEL_CONFIG['response_dim'],
        env_dim=MODEL_CONFIG['env_dim'],
        trans_dim=MODEL_CONFIG['trans_dim'],
        num_heads=MODEL_CONFIG['num_heads'],
        ff_hidden_dim=MODEL_CONFIG['ff_hidden_dim'],
        conv_hidden_dim=MODEL_CONFIG['conv_hidden_dim'],
        kernel_size=MODEL_CONFIG['kernel_size'],
        dropout=MODEL_CONFIG['dropout'],
        n_steps=MODEL_CONFIG['n_steps'],
        lag=MODEL_CONFIG['lag'],
        m=MODEL_CONFIG['m']
    ).to(device)
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"✓ Model loaded successfully: {MODEL_PATH}")
    except Exception as e:
        print(f"⚠ Failed to load model: {e}")
        raise
    
    # Load scalers
    try:
        with open(SCALER_ALL_PATH, 'rb') as f:
            scaler_all = pickle.load(f)
        print(f"✓ All data scaler loaded successfully: {SCALER_ALL_PATH}")
    except Exception as e:
        print(f"⚠ Failed to load scaler: {e}")
        raise
    
    try:
        with open(SCALER_RESPONSE_PATH, 'rb') as f:
            scaler_response = pickle.load(f)
        print(f"✓ Response scaler loaded successfully: {SCALER_RESPONSE_PATH}")
    except Exception as e:
        print(f"⚠ Failed to load response scaler: {e}")
        raise


def predict_crack(df, target_time=None, predict_steps=None):
    """
    Make predictions using the model
    
    Parameters:
        df: Data DataFrame
        target_time: Target prediction time (datetime object), if None, use the last time point of df
        predict_steps: Number of prediction time steps, if None, use MODEL_CONFIG['n_steps'] (default 6 steps)
    """
    global model, scaler_all, scaler_response, device
    
    if model is None or scaler_all is None or scaler_response is None:
        raise ValueError("Model or scalers not loaded")
    
    # Check if data is sufficient
    min_required = MODEL_CONFIG['lag'] + MODEL_CONFIG['m']
    if len(df) < min_required:
        raise ValueError(f"Insufficient data: need at least {min_required} time steps, currently have {len(df)}")
    
    # Determine prediction baseline time point
    if target_time is None:
        # Use the last time point for prediction
        current_index = len(df) - 1
        base_time = df.index[-1]
    else:
        # Find the nearest time point before (inclusive) target_time
        eligible_times = df.index[df.index <= target_time]
        if eligible_times.empty:
            raise ValueError(f"Cannot find data point before target_time {target_time}")
        base_time = eligible_times[-1]
        current_index = df.index.get_loc(base_time)
    
    # Determine prediction steps
    if predict_steps is None:
        predict_steps = MODEL_CONFIG['n_steps']
    else:
        predict_steps = min(predict_steps, MODEL_CONFIG['n_steps'])  # Cannot exceed model's maximum output steps
    
    # Check if there is sufficient non-null data
    required_cols = df.columns.tolist()
    df_filled = df.copy()
    
    # Handle missing values with forward fill and backward fill
    df_filled = df_filled.ffill().bfill()
    
    # If there are still missing values, fill with 0
    df_filled = df_filled.fillna(0)
    
    # Normalize data (pass DataFrame to avoid warnings)
    df_normalized = scaler_all.transform(df_filled)
    
    # Extract input data (df_normalized is numpy array)
    x_response = df_normalized[current_index-MODEL_CONFIG['m']+1:current_index+1, 4:7]  # Crack data (columns 4-6)
    x_env = df_normalized[current_index-MODEL_CONFIG['lag']+1:current_index+1, 15:17]  # Environmental data (columns 15-16)
    x_cat = df_normalized[current_index-MODEL_CONFIG['m']+1:current_index+1, :15]  # All structural responses (columns 0-14)
    
    # Convert to tensor and add batch dimension
    x_response = torch.tensor(x_response, dtype=torch.float32).unsqueeze(0).to(device)
    x_env = torch.tensor(x_env, dtype=torch.float32).unsqueeze(0).to(device)
    x_cat = torch.tensor(x_cat, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Model inference
    with torch.no_grad():
        output = model(x_response, x_env, x_cat)  # [1, n_steps, response_dim]
    
    # Inverse normalization
    output_numpy = output.cpu().numpy().reshape(-1, MODEL_CONFIG['response_dim'])
    output_original = scaler_response.inverse_transform(output_numpy)
    predictions = output_original.reshape(1, -1, MODEL_CONFIG['response_dim'])[0]
    
    # Generate prediction time points (only use first predict_steps time steps)
    # Predictions start 10 minutes after base_time
    prediction_times = [base_time + timedelta(minutes=10*(i+1)) for i in range(predict_steps)]
    
    # Build prediction results
    result = {
        'predictions': [],
        'timestamp': int(time.time()),
        'base_time': base_time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    for i, pred_time in enumerate(prediction_times):
        pred_point = {
            'time': pred_time.strftime("%Y-%m-%d %H:%M:%S"),
            'timestamp': int(pred_time.timestamp()),
            'crack_1': float(predictions[i][0]),
            'crack_2': float(predictions[i][1]),
            'crack_3': float(predictions[i][2])
        }
        result['predictions'].append(pred_point)
    
    return result


# ==========================================
# Scheduled Tasks
# ==========================================

def update_prediction_storage(predictions_result):
    """Update prediction storage, overwrite if time is the same"""
    global prediction_storage
    
    for pred_point in predictions_result['predictions']:
        time_key = pred_point['time']
        # If time is the same, overwrite old data with new data
        prediction_storage[time_key] = pred_point.copy()
    
    print(f"✓ Prediction storage updated, currently storing {len(prediction_storage)} time points")


def run_prediction(target_time=None, predict_steps=None):
    """
    Execute prediction task
    
    Parameters:
        target_time: Target prediction time (datetime object), if None, predict current time
        predict_steps: Number of prediction time steps, if None, use MODEL_CONFIG['n_steps'] (default 6 steps)
    """
    global latest_predictions, prediction_storage
    
    try:
        if target_time:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting prediction task (target time: {target_time.strftime('%Y-%m-%d %H:%M:%S')})...")
        else:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting prediction task...")
        
        # Fetch sensor data
        sensor_data = fetch_all_sensor_data()
        
        # Convert to DataFrame
        df = process_sensor_data_to_dataframe(sensor_data)
        print(f"DataFrame shape: {df.shape}")
        print(f"Data time range: {df.index[0]} to {df.index[-1]}")
        
        # Execute prediction
        predictions = predict_crack(df, target_time=target_time, predict_steps=predict_steps)
        latest_predictions = predictions
        
        # Update prediction storage (overwrite old data for same time)
        update_prediction_storage(predictions)
        
        print(f"✓ Prediction completed, generated {len(predictions['predictions'])} time points")
        if len(predictions['predictions']) > 0:
            print(f"Prediction time range: {predictions['predictions'][0]['time']} to {predictions['predictions'][-1]['time']}")
        
    except Exception as e:
        print(f"✗ Prediction task failed: {e}")
        import traceback
        traceback.print_exc()


def predict_past_12_hours():
    """Predict past 12 hours of data (backward prediction)"""
    print("\n" + "="*60)
    print("Starting prediction for past 12 hours of data...")
    print("="*60)
    
    now = datetime.now()
    # Round to 10 minutes
    now_minutes = now.minute
    rounded_minutes = ((now_minutes // 10) + (1 if now_minutes % 10 >= 5 else 0)) * 10
    if rounded_minutes >= 60:
        rounded_minutes = 0
        now = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        now = now.replace(minute=rounded_minutes, second=0, microsecond=0)
    
    # Calculate start time for past 12 hours (one time point per 10 minutes)
    start_time = now - timedelta(hours=12)
    
    # Generate all time points that need prediction (one per 10 minutes, from 12 hours ago to now)
    time_points = []
    current = start_time
    while current <= now:
        time_points.append(current)
        current += timedelta(minutes=10)
    
    print(f"Need to predict {len(time_points)} time points")
    print(f"Time range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Fetch sensor data
    sensor_data = fetch_all_sensor_data()
    df = process_sensor_data_to_dataframe(sensor_data)
    
    # Predict for each time point (backward prediction)
    success_count = 0
    fail_count = 0
    
    for i, target_time in enumerate(time_points):
        try:
            # Only predict data before target time (backward prediction)
            if target_time >= df.index[-1]:
                continue  # Skip future time points, these will be predicted in scheduled tasks
            
            # For each historical time point, predict its future 6 time steps
            predictions = predict_crack(df, target_time=target_time, predict_steps=MODEL_CONFIG['n_steps'])
            update_prediction_storage(predictions)
            success_count += 1
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(time_points)} ({success_count} succeeded, {fail_count} failed)")
        except Exception as e:
            fail_count += 1
            print(f"Failed to predict time point {target_time.strftime('%Y-%m-%d %H:%M:%S')}: {e}")
    
    print(f"\nPast 12 hours prediction completed: {success_count} succeeded, {fail_count} failed")
    print(f"Currently storing {len(prediction_storage)} time points")
    print("="*60 + "\n")


def schedule_periodic_prediction():
    """Execute prediction every 10 minutes (predict next 1 hour: 6 time steps)"""
    # Execute prediction for current time (predict next 1 hour: 6 time steps)
    run_prediction(predict_steps=MODEL_CONFIG['n_steps'])
    
    # Calculate next 10-minute time point
    now = datetime.now()
    minutes = now.minute
    # Calculate next 10-minute multiple
    next_minutes = ((minutes // 10) + 1) * 10
    if next_minutes >= 60:
        next_minutes = 0
        next_time = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    else:
        next_time = now.replace(minute=next_minutes, second=0, microsecond=0)
    
    delay = (next_time - now).total_seconds()
    
    print(f"Next prediction will execute at {next_time.strftime('%Y-%m-%d %H:%M:%S')} ({delay/60:.1f} minutes later)")
    
    timer = Timer(delay, schedule_periodic_prediction)
    timer.daemon = True
    timer.start()


# ==========================================
# FastAPI
# ==========================================

app = FastAPI(title="Crack Meter Prediction Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/api/predictions/crack')
def get_predictions():
    """Get prediction results: past 12 hours + future 1 hour"""
    global prediction_storage, latest_predictions
    
    if not prediction_storage:
        return JSONResponse(
            status_code=404,
            content={
                'success': False,
                'message': 'No prediction data available, please try again later'
            }
        )
    
    # Get current time
    now = datetime.now()
    
    # Calculate time range: past 12 hours to future 1 hour
    past_time = now - timedelta(hours=12)
    future_time = now + timedelta(hours=1)
    
    # Collect all predictions within the time range
    predictions_list = []
    for pred_point in prediction_storage.values():
        pred_time = datetime.strptime(pred_point['time'], "%Y-%m-%d %H:%M:%S")
        # Include predictions from past 12 hours to future 1 hour
        if past_time <= pred_time <= future_time:
            predictions_list.append(pred_point)
    
    if not predictions_list:
        return JSONResponse(
            status_code=404,
            content={
                'success': False,
                'message': 'No prediction data available for the requested time range, please try again later'
            }
        )
    
    # Sort by time
    predictions_list.sort(key=lambda x: x['timestamp'])
    
    return {
        'success': True,
        'data': {
            'predictions': predictions_list,
            'count': len(predictions_list),
            'timestamp': int(time.time())
        }
    }


@app.post('/api/predictions/crack/force')
def force_predict():
    """Force execute a prediction"""
    try:
        run_prediction()
        if latest_predictions is None:
            return JSONResponse(
                status_code=500,
                content={
                    'success': False,
                    'message': 'Prediction execution failed'
                }
            )
        
        return {
            'success': True,
            'data': latest_predictions
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'message': str(e)
            }
        )


@app.get('/api/health')
def health_check():
    """Health check"""
    return {
        'status': 'ok',
        'model_loaded': model is not None,
        'has_predictions': latest_predictions is not None,
        'prediction_storage_count': len(prediction_storage)
    }


@app.get('/api/model/metrics')
def get_model_metrics():
    """Get model training metrics (R², RMSE, MAE from training phase)"""
    return {
        'success': True,
        'data': {
            'r2_score': 0.88,
            'rmse': 0.25,
            'mae': 0.18,
            'description': 'Model performance metrics from training phase'
        }
    }


@app.get('/api/predictions/accuracy')
def get_prediction_accuracy():
    """Get real-time prediction accuracy by matching predictions with historical true values"""
    global prediction_storage
    
    try:
        # Fetch current sensor data to get true values
        sensor_data = fetch_all_sensor_data()
        df = process_sensor_data_to_dataframe(sensor_data)
        
        # Get current time
        now = datetime.now()
        past_time = now - timedelta(hours=12)  # Look back 12 hours for matched pairs
        
        # Store predictions by base_time (prediction batch)
        prediction_batches = {}  # {base_time_str: [pred_points]}
        
        # Group predictions by their original base_time
        # Note: We need to store base_time with each prediction point
        # For now, we'll estimate base_time from prediction time (pred_time - prediction_step * 10min)
        
        matched_pairs = []  # List of {pred_time, pred_value, true_value, sensor_id}
        
        # Match predictions with true values
        for pred_point in prediction_storage.values():
            pred_time_str = pred_point['time']
            pred_time = datetime.strptime(pred_time_str, "%Y-%m-%d %H:%M:%S")
            
            # Only process predictions in the past (can be validated)
            if pred_time <= now:
                # Find corresponding true value from historical data
                # Match within 2 minutes for accuracy
                matching_times = [t for t in df.index if abs((t - pred_time).total_seconds()) < 120]
                
                if matching_times:
                    true_time = matching_times[0]
                    
                    # Get true values for all 3 crack sensors
                    for sensor_idx in range(3):
                        sensor_col = f'crack_{sensor_idx+1}'
                        if sensor_col in df.columns:
                            true_value = df.loc[true_time, sensor_col]
                            pred_value = pred_point.get(f'crack_{sensor_idx+1}')
                            
                            if true_value is not None and not pd.isna(true_value) and pred_value is not None:
                                matched_pairs.append({
                                    'timestamp': int(pred_time.timestamp()),
                                    'time': pred_time_str,
                                    'sensor_id': sensor_idx + 1,
                                    'predicted': float(pred_value),
                                    'true': float(true_value),
                                    'error': float(abs(true_value - pred_value))
                                })
        
        # Calculate metrics for matched pairs
        if len(matched_pairs) > 0:
            # Group by sensor
            sensors = {1: [], 2: [], 3: []}
            for pair in matched_pairs:
                sensors[pair['sensor_id']].append(pair)
            
            # Calculate metrics per sensor
            metrics_per_sensor = {}
            for sensor_id, pairs in sensors.items():
                if len(pairs) >= 2:  # Need at least 2 points for R²
                    true_vals = [p['true'] for p in pairs]
                    pred_vals = [p['predicted'] for p in pairs]
                    
                    # Calculate R²
                    mean_true = sum(true_vals) / len(true_vals)
                    ss_res = sum((true_vals[i] - pred_vals[i])**2 for i in range(len(true_vals)))
                    ss_tot = sum((true_vals[i] - mean_true)**2 for i in range(len(true_vals)))
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Calculate RMSE
                    rmse = (ss_res / len(true_vals)) ** 0.5
                    
                    # Calculate MAE
                    mae = sum(abs(true_vals[i] - pred_vals[i]) for i in range(len(true_vals))) / len(true_vals)
                    
                    metrics_per_sensor[sensor_id] = {
                        'r2': r2,
                        'rmse': rmse,
                        'mae': mae,
                        'sample_count': len(pairs)
                    }
            
            # Calculate average metrics across all sensors
            if metrics_per_sensor:
                avg_r2 = sum(m['r2'] for m in metrics_per_sensor.values()) / len(metrics_per_sensor)
                avg_rmse = sum(m['rmse'] for m in metrics_per_sensor.values()) / len(metrics_per_sensor)
                avg_mae = sum(m['mae'] for m in metrics_per_sensor.values()) / len(metrics_per_sensor)
                total_samples = sum(m['sample_count'] for m in metrics_per_sensor.values())
            else:
                avg_r2 = avg_rmse = avg_mae = None
                total_samples = 0
        else:
            metrics_per_sensor = {}
            avg_r2 = avg_rmse = avg_mae = None
            total_samples = 0
        
        return {
            'success': True,
            'data': {
                'average': {
                    'r2': avg_r2,
                    'rmse': avg_rmse,
                    'mae': avg_mae,
                    'sample_count': total_samples
                },
                'per_sensor': metrics_per_sensor,
                'matched_pairs_count': len(matched_pairs),
                'timestamp': int(time.time())
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'message': f'Error calculating accuracy: {str(e)}'
            }
        )


# ==========================================
# Main Program
# ==========================================

if __name__ == '__main__':
    print("=" * 60)
    print("Crack Meter Prediction Service Starting")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    load_model_and_scalers()
    
    # Predict past 12 hours of data
    print("\nExecuting past 12 hours prediction...")
    predict_past_12_hours()
    
    # Start scheduled task (execute every 10 minutes, predict next 1 hour)
    print("\nStarting scheduled task (execute every 10 minutes, predict next 1 hour)...")
    schedule_periodic_prediction()
    
    # Start FastAPI service
    print("\nStarting FastAPI service...")
    print("API address: http://localhost:5000")
    print("Prediction endpoint: GET /api/predictions/crack")
    print("Force prediction: POST /api/predictions/crack/force")
    
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)

