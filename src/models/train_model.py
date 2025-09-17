from prophet_model import train_prophet
from lstm_model import train_lstm
from tqdm import tqdm
import time

if __name__ == "__main__":
    # ----------------------------
    # Prophet training
    # ----------------------------
    print("⏳ Starting Prophet training...")
    for _ in tqdm(range(1), desc="Prophet Progress"):
        prophet_model, prophet_5min, prophet_hourly = train_prophet(
            processed_file="data/preprocessed_dataset.csv",
            output_json_5min="outputs/forecast_prophet_5min.json",
            output_json_hourly="outputs/forecast_prophet_hourly.json",
        )
        time.sleep(0.1)

    print("✅ Prophet training completed!\n")

    # ----------------------------
    # LSTM training
    # ----------------------------
    print("⏳ Starting LSTM training...")
    for _ in tqdm(range(1), desc="LSTM Progress"):
        lstm_model, lstm_5min, lstm_hourly = train_lstm(
            processed_file="data/preprocessed_dataset.csv",
            output_json_5min="outputs/forecast_lstm_5min.json",
            output_json_hourly="outputs/forecast_lstm_hourly.json"
        )
        time.sleep(0.1)

    print("✅ LSTM training completed!")