from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
import joblib
from fastapi import Form
from io import StringIO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler (replace with actual timestamp)
MODEL_PATH = "../models/final_xgboost_model.pkl"
SCALER_PATH = "../models/final_xgboost_scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise Exception(f"Error loading model or scaler: {str(e)}")

# Column names for the 29-column raw input (excluding 'class')
COLUMNS_29 = ['type', 'sendtime', 'sender', 'senderpseudo', 'messageid', 'posx', 'posy', 'posz', 'posx_n', 'posy_n', 'posz_n',
              'spdx', 'spdy', 'spdz', 'spdx_n', 'spdy_n', 'spdz_n', 'aclx', 'acly', 'aclz', 'aclx_n', 'acly_n', 'aclz_n',
              'hedx', 'hedy', 'hedz', 'hedx_n', 'hedy_n', 'hedz_n']

# Features to select (15 features)
FEATURES_15 = ['sendtime', 'senderpseudo', 'posx', 'posy', 'posx_n', 'spdx', 'spdy', 'spdx_n', 'spdy_n', 'aclx', 'acly', 'hedx', 'hedy', 'hedx_n', 'hedy_n']

# Base features for feature engineering (8 features)
FEATURES_8 = ['senderpseudo', 'posx', 'posy', 'posx_n', 'spdx', 'spdy', 'hedy', 'hedx_n']

# Feature engineering function
def engineer_advanced_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    print("Starting advanced feature engineering...")
    
    # 1. Inter-packet arrival time
    X['sender_sequence'] = X.groupby('senderpseudo').cumcount()
    X['inter_arrival_time'] = X.groupby('senderpseudo')['sender_sequence'].diff().fillna(1.0)
    
    # 2. Speed and acceleration features
    X['speed_magnitude'] = np.sqrt(X['spdx']**2 + X['spdy']**2)
    X['acceleration_x'] = X.groupby('senderpseudo')['spdx'].diff().fillna(0)
    X['acceleration_y'] = X.groupby('senderpseudo')['spdy'].diff().fillna(0)
    X['acceleration_magnitude'] = np.sqrt(X['acceleration_x']**2 + X['acceleration_y']**2)
    
    # 3. Position change features
    X['position_change_x'] = X.groupby('senderpseudo')['posx'].diff().fillna(0)
    X['position_change_y'] = X.groupby('senderpseudo')['posy'].diff().fillna(0)
    X['position_change_magnitude'] = np.sqrt(X['position_change_x']**2 + X['position_change_y']**2)
    
    # 4. Heading features
    X['heading_change'] = X.groupby('senderpseudo')['hedy'].diff().fillna(0)
    X['heading_magnitude'] = np.sqrt(X['hedx_n']**2 + X['hedy']**2)
    
    # 5. Behavioral consistency features
    X['speed_consistency'] = X.groupby('senderpseudo')['speed_magnitude'].transform('std').fillna(0)
    X['position_consistency'] = X.groupby('senderpseudo')['position_change_magnitude'].transform('std').fillna(0)
    
    # 6. Temporal features
    X['hour'] = (X['sender_sequence'] % 24)
    X['night_hours'] = X['hour'].between(22, 5, inclusive="left").astype(int)
    
    # 7. Interaction features
    X['speed_position_interaction'] = X['speed_magnitude'] * X['position_change_magnitude']
    X['inter_arrival_speed_ratio'] = X['inter_arrival_time'] / (X['speed_magnitude'] + 1e-6)
    
    # Fill any remaining NaN values
    X = X.fillna(0)
    
    print(f"Feature engineering completed. New shape: {X.shape}")
    return X

@app.post("/predict")
async def predict_csv(file: UploadFile = File(...)):
    """
    Predict on CSV file upload only
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")
        
        # Read CSV file
        data = pd.read_csv(file.file)
        
        # Check if required columns are present (expecting 29 columns)
        if len(data.columns) != 29 or not all(col in data.columns for col in COLUMNS_29):
            raise ValueError(f"CSV must contain exactly 29 columns matching the dataset structure: {COLUMNS_29}")
        
        # Step 1: Select 15 features
        data_15 = data[FEATURES_15].copy()
        
        # Step 2: Select 8 base features for feature engineering
        data_8 = data_15[FEATURES_8].copy()
        
        # Step 3: Apply feature engineering
        data_engineered = engineer_advanced_features(data_8)
        
        # Step 4: Standardization on the engineered features
        data_scaled = scaler.transform(data_engineered)
        
        # Step 5: Predict
        proba = model.predict_proba(data_scaled)
        predictions = np.argmax(proba, axis=1).tolist()
        probabilities = proba.tolist()
        
        return {
            "predictions": predictions, 
            "probabilities": probabilities,
            "total_samples": len(predictions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_single")
async def predict_single(data: str = Form(...)):
    """
    Predict on a single row of comma-separated values
    Enhanced version with better error handling
    """
    try:
        # Clean the input string
        data = data.strip()
        
        # Convert the string to a DataFrame with the correct columns
        df = pd.read_csv(StringIO(data), header=None, names=COLUMNS_29)
        
        # Validate that we have exactly 29 values
        if len(df.columns) != 29:
            raise ValueError(f"Expected 29 values, got {len(df.columns)}")
        
        # Step 1: Select 15 features
        data_15 = df[FEATURES_15].copy()
        
        # Step 2: Select 8 base features for feature engineering
        data_8 = data_15[FEATURES_8].copy()
        
        # Step 3: Apply feature engineering
        data_engineered = engineer_advanced_features(data_8)
        
        # Step 4: Standardization on the engineered features
        data_scaled = scaler.transform(data_engineered)
        
        # Step 5: Predict
        proba = model.predict_proba(data_scaled)
        predictions = np.argmax(proba, axis=1).tolist()
        probabilities = proba.tolist()
        
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "total_samples": len(predictions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")



@app.get("/health")
async def health_check():
    return {"status": "API is running"}
