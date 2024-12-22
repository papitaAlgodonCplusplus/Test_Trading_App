import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pandas_ta as ta

class TemporalFusionTransformer:
    def __init__(self, data_path, window_size=100, batch_size=32, num_classes=3, hidden_size=64, num_heads=4, learning_rate=0.001, epochs=50):
        self.data_path = data_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.scaler = StandardScaler()
        self.model = None
        self.dataloader = None
        
    def load_and_preprocess_data(self, return_dataset=False):
        df = pd.read_csv(self.data_path)
        df = self.add_technical_indicators(df)
        df = df.dropna().reset_index(drop=True)
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'Signal', 'Stochastic_K', 'Stochastic_D', 'SMA', 'Upper_Band', 'Lower_Band', 'RSI']
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        df['Label'] = self.calculate_trend_labels(df)
        dataset = self.TimeSeriesDataset(df, self.window_size, feature_cols)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        if return_dataset:
            return dataset
        
    def add_technical_indicators(self, df):
        macd = df.ta.macd(close='Close', fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        df['Signal'] = macd['MACDs_12_26_9']
        stoch = df.ta.stoch(high='High', low='Low', close='Close', k=14, d=3)
        df['Stochastic_K'] = stoch['STOCHk_14_3_3']
        df['Stochastic_D'] = stoch['STOCHd_14_3_3']
        df['SMA'] = df.ta.sma(close='Close', length=20)
        bbands = df.ta.bbands(close='Close', length=20, std=2)
        df['Upper_Band'] = bbands['BBU_20_2.0']
        df['Lower_Band'] = bbands['BBL_20_2.0']
        df['RSI'] = df.ta.rsi(close='Close', length=14)
        return df
    
    def calculate_trend_labels(self, df, threshold=0.0001):
        labels = []
        for i in range(len(df) - self.window_size):
            future_close = df['Close'].iloc[i + self.window_size]
            current_close = df['Close'].iloc[i]
            change = (future_close - current_close) / current_close
            if change > threshold:
                labels.append(1)
            elif change < -threshold:
                labels.append(2)
            else:
                labels.append(0)
        labels.extend([None] * self.window_size)
        return labels
    
    class TimeSeriesDataset(Dataset):
        def __init__(self, df, window_size, feature_cols):
            valid_indices = df['Label'].notna()
            self.data = df[feature_cols].values[valid_indices]
            self.labels = df['Label'].values[valid_indices]
            self.window_size = window_size
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            X = self.data[idx:idx + self.window_size]
            if len(X) < self.window_size:
                X = np.pad(X, ((0, self.window_size - len(X)), (0, 0)), mode='constant')
            y = self.labels[idx]
            return torch.FloatTensor(X), torch.LongTensor([y])
        
    class TFT(nn.Module):
        def __init__(self, input_size, num_classes, hidden_size, num_heads):
            super().__init__()
            self.input_proj = nn.Linear(input_size, hidden_size)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads),
                num_layers=3
            )
            self.fc = nn.Linear(hidden_size, num_classes)
            
        def forward(self, x):
            x = self.input_proj(x)
            x = x.permute(1, 0, 2)  
            x = self.transformer(x)
            x = x[-1, :, :]  
            return self.fc(x)
        
    def initialize_model(self):
        self.model = self.TFT(input_size=13, num_classes=self.num_classes, hidden_size=self.hidden_size, num_heads=self.num_heads)
        
    def train_model(self, version=None):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_val_loss = float('inf')
        best_model_state = None
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            train_size = int(0.8 * len(self.dataloader.dataset))
            val_size = len(self.dataloader.dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(self.dataloader.dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            val_loss = 0
            self.model.eval()
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch.view(-1))
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
        if best_model_state is not None:
            if version is not None:
                torch.save(best_model_state, f'analyzers/models/tft_model_{version}.pth')
            else:
                torch.save(best_model_state, 'analyzers/models/tft_model.pth')
            
    def evaluate_model(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in self.dataloader:
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch.view(-1)).sum().item()
        print(f"Accuracy: {100 * correct / total:.2f}%")
        
    def run(self, version=None):
        self.load_and_preprocess_data()
        self.initialize_model()
        self.train_model(version)
        self.evaluate_model()
        
    def predict(self, data_path, version=None):
        self.data_path = data_path
        if self.model is None:
            self.initialize_model()
        if version is not None:
            self.model.load_state_dict(torch.load(f'analyzers/models/tft_model_{version}.pth', weights_only=True))
        else:
            self.model.load_state_dict(torch.load('analyzers/models/tft_model.pth', weights_only=True))
        self.model.eval()
        dataset = self.load_and_preprocess_data(True)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            outputs = self.model(next(iter(self.dataloader))[0])
            _, predicted = torch.max(outputs, 1)
            if predicted.item() == 1:
                return 1, outputs
            elif predicted.item() == 2:
                return -1, outputs
            else:
                return 0, outputs