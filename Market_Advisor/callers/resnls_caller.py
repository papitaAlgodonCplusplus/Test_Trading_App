import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import torch
from analyzers.resnls import ResNLS

def indicate():
    # Load data
    data = pd.read_csv('data/real_time_data.csv')
    prices = pd.DataFrame(data)

    # Preprocessing
    seq_length = 5
    data_values = prices[['Close', 'Volume']].values.astype('float32')
    data_mean = data_values.mean(axis=0)
    data_std = data_values.std(axis=0)
    data_normalized = (data_values - data_mean) / data_std

    # Create sequences and targets
    X, y = [], []
    for i in range(len(data_normalized) - seq_length):
        X.append(data_normalized[i:i + seq_length])
        y.append(data_normalized[i + seq_length, 0])  # Predict 'Close'

    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    model = ResNLS(input_channels=2, seq_length=seq_length, hidden_size=64)
    best_model_path = os.path.join(os.path.dirname(__file__), '..', 'analyzers', 'models', 'resnls_best.pth')
    model.load_state_dict(torch.load(best_model_path))
    with torch.no_grad():
        predictions = model(X.permute(0, 2, 1)).squeeze().numpy()
    
    last_predicted_price = predictions[-1]
    last_actual_price = predictions[0]
    print(f"Last actual price: {last_actual_price}, Last predicted price: {last_predicted_price}")
    if last_predicted_price > last_actual_price:
        return 1
    elif last_predicted_price < last_actual_price:
        return -1
    else:
        return 0

if __name__ == "__main__":
    indicate()
