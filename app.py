import torch
import torch.nn as nn
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, Response, json
import logging
from datetime import datetime
import time

app = Flask(__name__)

# Configure basic logging without JSON formatting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the optimized model with the correct architecture
class EnhancedBiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, dropout):
        super(EnhancedBiLSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size * 2)  # *2 for bidirectional and 2 timeframes

    def forward(self, input_seq):
        h_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size).to(device)
        c_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size).to(device)
        
        lstm_out, _ = self.lstm(input_seq, (h_0, c_0))
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# Initialize and load the model with pre-trained weights
model = EnhancedBiLSTMModel(input_size=1, hidden_layer_size=115, output_size=1, num_layers=2, dropout=0.3).to(device)
model.load_state_dict(torch.load("enhanced_bilstm_model.pth", map_location=device))
model.eval()

# Function to fetch historical data from Binance
def get_binance_data(symbol="ETHUSDT", interval="1m", limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    response.raise_for_status()  # Ensure that we raise an error if the request fails
    return response.json()

# Dynamic scaling factor based on token and time of day
def get_dynamic_scaling_factor(token, current_time):
    hour = current_time.hour
    
    # Token-specific scaling logic
    scaling_factors = {
        'BTCUSDT': {
            'morning': 1.05,
            'afternoon': 1.03,
            'evening': 1.04,
            'night': 1.02
        },
        'ETHUSDT': {
            'morning': 1.04,
            'afternoon': 1.02,
            'evening': 1.03,
            'night': 1.01
        },
        'BNBUSDT': {
            'morning': 1.03,
            'afternoon': 1.02,
            'evening': 1.01,
            'night': 1.02
        },
        'SOLUSDT': {
            'morning': 1.02,
            'afternoon': 1.03,
            'evening': 1.04,
            'night': 1.03
        },
        'ARBUSDT': {
            'morning': 1.01,
            'afternoon': 1.02,
            'evening': 1.03,
            'night': 1.01
        }
    }
    
    # Default scaling factor
    default_factor = 1.02
    
    # Determine the time period
    if 0 <= hour < 6:
        period = 'night'
    elif 6 <= hour < 12:
        period = 'morning'
    elif 12 <= hour < 18:
        period = 'afternoon'
    else:
        period = 'evening'
    
    # Get the scaling factor for the token and time period
    return scaling_factors.get(token, {}).get(period, default_factor)

@app.route("/inference/<string:token>")
def get_inference(token):
    symbol_map = {
        'ETH': 'ETHUSDT',
        'BTC': 'BTCUSDT',
        'BNB': 'BNBUSDT',
        'SOL': 'SOLUSDT',
        'ARB': 'ARBUSDT'
    }

    token = token.upper()
    if token not in symbol_map:
        return Response(json.dumps({"error": "Unsupported token"}), status=400, mimetype='application/json')

    symbol = symbol_map[token]

    try:
        # Fetch and preprocess data
        start_time = time.time()
        data = get_binance_data(symbol=symbol)
        df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume", "close_time",
                                         "quote_asset_volume", "number_of_trades",
                                         "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])

        df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
        df = df[["close_time", "close"]].tail(20 if symbol != 'BTCUSDT' and symbol != 'SOLUSDT' else 10)
        df.columns = ["date", "price"]
        df["price"] = df["price"].astype(float)

        # Log the current price and the timestamp
        current_price = df.iloc[-1]["price"]
        current_time = df.iloc[-1]["date"]
        logger.info(f"Current Price: {current_price} at {current_time}")

        # Normalize and prepare data for model
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(df['price'].values.reshape(-1, 1))

        seq = torch.FloatTensor(scaled_data).view(1, -1, 1).to(device)

        # Perform inference
        with torch.no_grad():
            y_pred = model(seq)

        # Inverse transform the predictions to get the actual prices
        predicted_prices = scaler.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))

        # Apply dynamic scaling factor
        scaling_factor = get_dynamic_scaling_factor(symbol, current_time)
        predicted_prices *= scaling_factor

        # Select the appropriate prediction
        predicted_price = round(float(predicted_prices[0][0] if symbol in ['BTCUSDT', 'SOLUSDT'] else predicted_prices[1][0]), 2)

        # Log the prediction and performance time
        end_time = time.time()
        logger.info(f"Prediction: {predicted_price} (Computed in {end_time - start_time:.2f}s)")

        # Return only the predicted price in JSON response
        return Response(json.dumps(predicted_price), status=200, mimetype='application/json')
    
    except requests.RequestException as e:
        logger.error(f"Binance API request failed: {e}")
        return Response(json.dumps({"error": "Failed to retrieve data from Binance API"}), status=500, mimetype='application/json')
    except Exception as e:
        logger.error(f"An error occurred during inference: {e}")
        return Response(json.dumps({"error": "Internal server error"}), status=500, mimetype='application/json')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
