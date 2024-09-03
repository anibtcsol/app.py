import torch
import torch.nn as nn
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the attention mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.W = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.V = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        # Apply the attention mechanism
        scores = self.V(torch.tanh(self.W(lstm_output)))  # [batch_size, seq_length, 1]
        attention_weights = torch.softmax(scores, dim=1)  # [batch_size, seq_length, 1]
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # [batch_size, hidden_size * 2]
        return context_vector

# Define the enhanced model with LSTM and Attention
class EnhancedBiLSTMWithAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, dropout):
        super(EnhancedBiLSTMWithAttentionModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size * 2)  # *2 for bidirectional and 2 timeframes

    def forward(self, input_seq):
        h_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size).to(device)
        c_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size).to(device)
        
        lstm_out, _ = self.lstm(input_seq, (h_0, c_0))
        context_vector = self.attention(lstm_out)
        predictions = self.linear(context_vector)
        return predictions

# Function to fetch historical data from Binance
def get_binance_data(symbol="ETHUSDT", interval="1m", limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
        df = df[["close_time", "close"]]
        df.columns = ["date", "price"]
        df["price"] = df["price"].astype(float)
        return df
    except requests.RequestException as e:
        logger.error(f"Failed to retrieve data: {e}")
        raise

# Prepare the dataset
def prepare_dataset(symbols, sequence_length=10):
    all_data = []
    scaler_dict = {}
    
    for symbol in symbols:
        df = get_binance_data(symbol)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(df['price'].values.reshape(-1, 1))
        scaler_dict[symbol] = scaler  # Save scaler for each symbol
        
        for i in range(sequence_length, len(scaled_data) - 2):  # to account for the 20-minute prediction
            seq = scaled_data[i-sequence_length:i]
            label_10 = scaled_data[i+10] if i+10 < len(scaled_data) else scaled_data[-1]
            label_20 = scaled_data[i+20] if i+20 < len(scaled_data) else scaled_data[-1]
            label = torch.FloatTensor([label_10[0], label_20[0]])
            all_data.append((seq, label))
    
    # Split data into training and validation sets
    train_data, val_data = train_test_split(all_data, test_size=0.2, shuffle=True)
    return (train_data, val_data), scaler_dict

# Define the training process
def train_model(model, data, epochs=50, lr=0.001, sequence_length=10):
    model.to(device)
    train_data, val_data = data
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for seq, label in train_data:
            seq = torch.FloatTensor(seq).view(1, sequence_length, -1).to(device)
            label = label.view(1, -1).to(device)  # Ensure label has the shape [batch_size, 2]

            optimizer.zero_grad()
            y_pred = model(seq)
            loss = criterion(y_pred, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, label in val_data:
                seq = torch.FloatTensor(seq).view(1, sequence_length, -1).to(device)
                label = label.view(1, -1).to(device)
                y_pred = model(seq)
                loss = criterion(y_pred, label)
                val_loss += loss.item()

        logger.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss/len(train_data)}, Validation Loss: {val_loss/len(val_data)}')

    torch.save(model.state_dict(), "enhanced_bilstm_attention_model.pth")
    logger.info("Model trained and saved as enhanced_bilstm_attention_model.pth")

if __name__ == "__main__":
    # Define the model
    model = EnhancedBiLSTMWithAttentionModel(input_size=1, hidden_layer_size=115, output_size=1, num_layers=2, dropout=0.3)

    # Symbols to train on
    symbols = ['BNBUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ARBUSDT']

    # Prepare data
    data, scaler_dict = prepare_dataset(symbols)

    # Train the model
    train_model(model, data)
