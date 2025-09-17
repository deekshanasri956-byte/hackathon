import os
import json
import pandas as pd

def save_forecast_json(forecast_df, output_file="outputs/forecast.json", last_n=None):
    """
    Saves forecast DataFrame as JSON and detects peak hours (top 5%).

    Parameters:
    - forecast_df: pd.DataFrame with columns ['ds', 'yhat']
    - output_file: path to save JSON
    - last_n: int, number of last records to include in 'forecast'. 
              If None, include all rows.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Determine threshold for peak demand (top 5%)
    threshold = forecast_df['yhat'].quantile(0.95)
    peaks = forecast_df[forecast_df['yhat'] >= threshold]['ds'].astype(str).tolist()
    
    # Select last N rows for forecast
    if last_n:
        forecast_data = forecast_df[['ds', 'yhat']].tail(last_n)
    else:
        forecast_data = forecast_df[['ds', 'yhat']]
    
    output = {
        "forecast": forecast_data.to_dict(orient="records"),
        "peaks": peaks
    }
    
    # Save JSON
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Forecast JSON saved at {output_file}")