import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import xgboost as xgb

class LSTMPricePredictor:
    def __init__(self):
        self.model = None
        self.best_model = "analyzers/models/best_model.keras"
        self.scaler = MinMaxScaler()
        self.xgboost_model = None

    def build_model(self, input_shape):
        model = Sequential()
        
        # CNN layers
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        
        # Bidirectional LSTM layers
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(100)))
        model.add(Dropout(0.2))

        # Attention mechanism
        model.add(Dense(100, activation='relu'))
        attention_input = Input(shape=(input_shape[0], input_shape[1]))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def calculate_support_resistance(self, data, window=14):
        data['Support'] = data['Low'].rolling(window=window).min()
        data['Resistance'] = data['High'].rolling(window=window).max()
        return data

    def calculate_vwap(self, data):
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        return data

    def calculate_moving_averages(self, data):
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        return data

    def preprocess_data(self, data):
        data = self.calculate_support_resistance(data)
        data = self.calculate_vwap(data)
        data = self.calculate_moving_averages(data)
        data = data.dropna()
        scaled_data = self.scaler.fit_transform(data)
        return scaled_data, data

    def train(self, epochs, data_path):
        # Load data
        data = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Preprocess data
        scaled_data, _ = self.preprocess_data(data)
        X, y = [], []
        sequence_length = 60
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 3])  # Using 'Close' price as the target

        X, y = np.array(X), np.array(y)

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build model
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        # Define checkpoint
        os.makedirs(os.path.dirname(self.best_model), exist_ok=True)
        checkpoint = ModelCheckpoint(self.best_model, monitor='val_loss', save_best_only=True, mode='min')

        # Train LSTM model
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[checkpoint]
        )

        # Train XGBoost model
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        self.xgboost_model = xgb.XGBRegressor()
        self.xgboost_model.fit(
            X_train_flat, 
            y_train, 
            eval_set=[(X_val_flat, y_val)], 
        )
        
    def predict_future_prices(self, prices_df, steps_ahead=30):
        # Load the model if it exists
        model_path = "analyzers/models/best_model.keras"
        if os.path.exists(model_path):
            self.model = load_model(model_path)

        # Preprocess the input data
        data = prices_df[['Open', 'High', 'Low', 'Close', 'Volume']]
        scaled_data, _ = self.preprocess_data(data)

        # Prepare the input sequence
        input_sequence = scaled_data[-60:]  # Last 60 steps for prediction
        predictions = []

        for _ in range(steps_ahead):
            input_sequence_reshaped = input_sequence.reshape(1, input_sequence.shape[0], input_sequence.shape[1])
            lstm_predicted_value = self.model.predict(input_sequence_reshaped, verbose=0)[0][0]

            # Use XGBoost for final prediction
            xgboost_predicted_value = self.xgboost_model.predict(input_sequence_reshaped.reshape(1, -1))[0]

            # Combine predictions (e.g., weighted average)
            final_prediction = 0.7 * lstm_predicted_value + 0.3 * xgboost_predicted_value

            # Append prediction and update input sequence
            predictions.append(final_prediction)
            input_sequence = np.append(input_sequence, [[0] * (input_sequence.shape[1] - 1) + [final_prediction]], axis=0)
            input_sequence = input_sequence[1:]

        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        dummy_rest = np.zeros((predictions.shape[0], scaled_data.shape[1] - 1))
        predictions_full = np.concatenate((dummy_rest[:, :3], predictions, dummy_rest[:, 3:]), axis=1)
        future_prices = self.scaler.inverse_transform(predictions_full)[:, 3]  # Extract 'Close' prices

        return future_prices
