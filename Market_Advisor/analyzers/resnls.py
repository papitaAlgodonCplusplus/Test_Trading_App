import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = self.skip(x) if self.skip else x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)

class LSTMModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
        super(LSTMModule, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]

class ResNLS(nn.Module):
    def __init__(self, input_channels, seq_length, hidden_size, num_res_blocks=3, num_lstm_layers=2):
        super(ResNLS, self).__init__()
        self.res_blocks = nn.ModuleList([
            ResNetBlock(input_channels if i == 0 else hidden_size, hidden_size) for i in range(num_res_blocks)
        ])
        self.lstm = LSTMModule(hidden_size, hidden_size, num_lstm_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        for block in self.res_blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        x = self.lstm(x)
        return self.fc(x)

    def train_model(self, train_loader, val_loader, epochs, criterion, optimizer):
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self(x_batch.permute(0, 2, 1))
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    val_outputs = self(x_val.permute(0, 2, 1))
                    val_loss += criterion(val_outputs, y_val).item()

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}")

    def predict_future_prices(self, prices, steps_ahead=30):
        self.eval()
        data = prices[['Close', 'Volume']].values.astype(np.float32)
        seq_length = len(data)

        # Normalize data
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0)
        data_normalized = (data - data_mean) / data_std

        # Create sequences for prediction
        x_seq = torch.tensor(data_normalized[-seq_length:].T, dtype=torch.float32).unsqueeze(0)

        future_prices = []
        for step in range(steps_ahead):
            with torch.no_grad():
                pred = self(x_seq).item()

            # Denormalize prediction
            pred_denormalized = pred * data_std[0] + data_mean[0]
            future_prices.append(pred_denormalized)

            # Update sequence with predicted value
            next_step = np.array([pred, data_normalized[-1, 1]])  # Using same volume for simplicity
            next_step_normalized = (next_step - data_mean) / data_std
            x_seq = torch.cat((x_seq[:, :, 1:], torch.tensor(next_step_normalized).unsqueeze(0).unsqueeze(2)), dim=2)

        # Prepare future prices DataFrame
        future_prices_df = pd.DataFrame({
            'Step': range(1, steps_ahead + 1),
            'Predicted_Close': future_prices
        })

        return future_prices_df