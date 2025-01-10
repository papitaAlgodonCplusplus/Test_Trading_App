import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os

class PriceDataset(Dataset):
    def __init__(self, data, input_size, target_size):
        self.data = data
        self.input_size = input_size
        self.target_size = target_size

    def __len__(self):
        return len(self.data) - self.input_size - self.target_size + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.input_size]
        y = self.data[idx + self.input_size:idx + self.input_size + self.target_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class NBEATSModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NBEATSModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class NBEATS:
    def __init__(self, input_size=30, hidden_size=64, target_size=30):
        self.input_size = input_size
        self.target_size = target_size
        self.model = NBEATSModel(input_size, hidden_size, target_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.best_model_path = "analyzers/models/best_nbeats_model.keras"

    def train(self, epochs, data_path):
        # Load data
        data = pd.read_csv(data_path)
        data['Datetime'] = pd.to_datetime(data['Date'])
        data.set_index('Datetime', inplace=True)
        prices = data['Close'].values

        # Normalize data
        mean = prices.mean()
        std = prices.std()
        prices_normalized = (prices - mean) / std

        # Split data into training and validation sets
        split_idx = int(len(prices_normalized) * 0.8)
        train_prices = prices_normalized[:split_idx]
        val_prices = prices_normalized[split_idx:]

        # Create datasets and dataloaders
        train_dataset = PriceDataset(train_prices, self.input_size, self.target_size)
        val_dataset = PriceDataset(val_prices, self.input_size, self.target_size)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        self.model.train()
        best_val_loss = float('inf')

        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in train_dataloader:
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            train_loss = epoch_loss / len(train_dataloader)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_dataloader:
                    output = self.model(x)
                    loss = self.criterion(output, y)
                    val_loss += loss.item()
            val_loss /= len(val_dataloader)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not os.path.exists(os.path.dirname(self.best_model_path)):
                    os.makedirs(os.path.dirname(self.best_model_path))
                torch.save(self.model.state_dict(), self.best_model_path)

    def predict_future_prices(self, prices_df, steps_ahead=30):
        # Load the best model
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path))
        else:
            raise FileNotFoundError(f"Model file not found at {self.best_model_path}")

        # Prepare input data
        prices = prices_df['Close'].values
        mean = prices.mean()
        std = prices.std()
        prices_normalized = (prices - mean) / std

        # Get the most recent data
        input_data = prices_normalized[-self.input_size:]
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(input_tensor).squeeze(0).numpy()

        # Denormalize predictions
        future_prices = predictions * std + mean
        return future_prices

