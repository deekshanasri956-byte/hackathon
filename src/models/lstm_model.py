import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from utils import save_forecast_json

def create_sequences(data, seq_length=24):
    """
    Convert time-series data into sequences for LSTM.
    Each sequence has seq_length timesteps and predicts the next value.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_lstm(processed_file="data/processed_delhi_demand.csv",
               model_file="saved_models/lstm_model.h5",
               scaler_file="saved_models/demand_scaler.pkl",
               output_json="outputs/forecast_lstm.json",
               seq_length=24, epochs=30, batch_size=16):
    
    # ----------------------------
    # 1. Load processed dataset
    # ----------------------------
    data = pd.read_csv(processed_file, parse_dates=["datetime"])
    demand = data["demand"].values.reshape(-1,1)
    
    # ----------------------------
    # 2. Scale demand
    # ----------------------------
    scaler = MinMaxScaler()
    demand_scaled = scaler.fit_transform(demand)
    
    # Save scaler for inference
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(scaler, scaler_file)
    
    # ----------------------------
    # 3. Create sequences
    # ----------------------------
    X, y = create_sequences(demand_scaled, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # LSTM expects 3D input
    
    # ----------------------------
    # 4. Train/Test split
    # ----------------------------
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # ----------------------------
    # 5. Build LSTM model
    # ----------------------------
    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_length,1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    
    # ----------------------------
    # 6. Train model
    # ----------------------------
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size, verbose=1)
    
    # ----------------------------
    # 7. Save trained model
    # ----------------------------
    model.save(model_file)
    print(f"✅ LSTM model saved at {model_file}")
    
    # ----------------------------
    # 8. Forecast next 24 hours
    # ----------------------------
    last_seq = demand_scaled[-seq_length:]
    input_seq = np.expand_dims(last_seq, axis=0)
    forecast_scaled = []
    
    for _ in range(24):
        pred = model.predict(input_seq, verbose=0)
        forecast_scaled.append(pred[0,0])
        input_seq = np.append(input_seq[:,1:,:], [[pred]], axis=1)
    
    forecast_scaled = np.array(forecast_scaled).reshape(-1,1)
    forecast = scaler.inverse_transform(forecast_scaled)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        "ds": pd.date_range(start=data["datetime"].iloc[-1] + pd.Timedelta(hours=1),
                            periods=24, freq='H'),
        "yhat": forecast.flatten()
    })
    
    # ----------------------------
    # 9. Save forecast JSON
    # ----------------------------
    save_forecast_json(forecast_df, output_json)
    
    return model, forecast_df

# ----------------------------
# If run as script
# ----------------------------
if __name__ == "__main__":
    train_lstm()