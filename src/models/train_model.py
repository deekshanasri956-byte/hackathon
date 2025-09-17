from prophet_model import train_prophet
from lstm_model import train_lstm

if __name__ == "__main__":
    # Train Prophet
    train_prophet()

    # Train LSTM
    train_lstm()