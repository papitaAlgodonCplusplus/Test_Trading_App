import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense, Flatten, Dropout, BatchNormalization, MultiHeadAttention, LayerNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class PositionalEncoding(Layer):
    def __init__(self, position, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                    np.arange(d_model)[np.newaxis, :],
                                    d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)
    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]
        
class MarketAdviser:
    def __init__(self, model_dir='analyzers/models'):
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, 'market_adviser.h5')
        self.features = [
            'Open', 'High', 'Low', 'Close', 'Hour', 'DayOfWeek',
            'RollingMean_Close', 'RollingStd_Close',
            'RollingMean_High', 'RollingMean_Low',
            'PctChange_Close', 'High_Low_Range'
        ]
        self.model = None
        os.makedirs(self.model_dir, exist_ok=True)

    def preprocess_data(self, data, window=10):
        """
        Preprocess the dataset to make it suitable for training, including feature engineering.
        """
        data['Date'] = pd.to_datetime(data['Date'])
        data['Hour'] = data['Date'].dt.hour
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        data['RollingMean_Close'] = data['Close'].rolling(window=window).mean()
        data['RollingStd_Close'] = data['Close'].rolling(window=window).std()
        data['RollingMean_High'] = data['High'].rolling(window=window).mean()
        data['RollingMean_Low'] = data['Low'].rolling(window=window).mean()
        data['PctChange_Close'] = data['Close'].pct_change()
        data['High_Low_Range'] = data['High'] - data['Low']
        data.dropna(inplace=True)
        return data
    
    def calculate_profit_label(self, data, window=10):
        """
        Simulate profits for hypothetical actions to create the target label.
        - 1 if any of the next 5 days has a greater price than the current day.
        - 0 otherwise.
        """
        profit_labels = []
        for i in range(len(data) - window):
            current_price = data['Close'].iloc[i]
            future_prices = data['Close'].iloc[i + 1:i + 6]
            profit_labels.append(1 if (future_prices > current_price).any() else 0)
        profit_labels.extend([None] * window)
        data['Profit_Label'] = profit_labels
        data.dropna(subset=['Profit_Label'], inplace=True)
        data['Profit_Label'] = data['Profit_Label'].astype(int)
        return data
    
    def build_model(self, input_shape):
        """Build a robust neural network model with state-of-the-art enhancements."""
        model = Sequential([
            Dense(128, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  
        ])
        model.compile(optimizer=Adam(learning_rate=0.0005),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        return model
    
    def build_transformer_model(self, input_shape):
        """Build a Transformer-based neural network model."""
        inputs = Input(shape=input_shape)
        inputs = PositionalEncoding(input_shape[0], input_shape[1])(inputs)
        attention_output = MultiHeadAttention(num_heads=4, key_dim=input_shape[-1])(inputs, inputs)
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
        feed_forward = Dense(128, activation='relu')(attention_output)
        feed_forward = Dropout(0.3)(feed_forward)
        feed_forward = LayerNormalization(epsilon=1e-6)(feed_forward)
        flattened = Flatten()(feed_forward)
        outputs = Dense(1, activation='sigmoid')(flattened)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.0005),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        return model
    
    def train_model(self, dataset_path, epochs=100, batch_size=32, transformer=False, window_size=20,
                    version=None, window=10):
        """Train the model using the dataset and save it."""
        data = pd.read_csv(dataset_path)
        data = self.preprocess_data(data, window=window)
        data = self.calculate_profit_label(data, window=window)
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[self.features].iloc[i - window_size:i].values.flatten())
            y.append(data['Profit_Label'].iloc[i])
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        sequence_length = window_size
        feature_size = len(self.features)
        usable_samples = len(X) - (len(X) % sequence_length)
        if transformer:
            X = X[:usable_samples] 
            y = y[:usable_samples]
            X = np.array(X)
            y = np.array(y)
            if len(X) > 0:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_train = X_train.reshape((-1, sequence_length, feature_size))
            else:
                raise ValueError("Not enough samples to perform train-test split.")
            X_test = X_test.reshape((-1, sequence_length, feature_size))
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       
        if self.model is None:
            input_shape = (sequence_length, feature_size) if transformer else (X_train.shape[1],)
            if transformer:
                self.model = self.build_transformer_model(input_shape)
            else:
                self.model = self.build_model(input_shape)
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=max(1, int(epochs * 0.1)),
            restore_best_weights=True,
            min_delta=0.05,
            mode='max'
        )
        
        _ = self.model.fit(
            X_train, y_train, epochs=epochs, batch_size=batch_size,
            validation_data=(X_test, y_test), callbacks=[early_stopping]
        )
        if version is not None:
            self.model.save(f"analyzers/models\market_adviser_{version}.h5")
            print(f"analyzers/models\market_adviser_{version}.h5")
        else:
            self.model.save(self.model_path)
            print(f"Model trained and saved at {self.model_path}")
        self.is_transformer = transformer
        transformer_flag_path = os.path.join(self.model_dir, 'is_transformer.txt')
        with open(transformer_flag_path, 'w') as f:
            f.write(str(self.is_transformer))
        final_acc = self.model.evaluate(X_test, y_test, verbose=0)[1]
        print(f"Final Test Accuracy: {final_acc:.2f}")

    def load_model(self, version=None):
        """Load the trained model."""
        if os.path.exists(self.model_path):
            if version is not None:
                self.model = tf.keras.models.load_model(f"analyzers/models\market_adviser_{version}.h5")
                print(f"Model loaded from analyzers/models\market_adviser_{version}.h5")
            else:
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Model loaded from {self.model_path}")
            transformer_flag_path = os.path.join(self.model_dir, 'is_transformer.txt')
            if os.path.exists(transformer_flag_path):
                with open(transformer_flag_path, 'r') as f:
                    self.is_transformer = f.read() == 'True'
            else:
                self.is_transformer = False
            print("Model loaded.")
        else:
            raise FileNotFoundError("Model not found. Train the model first.")

    def predict_last_recommended_action(self, dataset_path, window_size=20, version=None):
        """Predict the recommended action for the last data point."""
        data = pd.read_csv(dataset_path)
        data = self.preprocess_data(data, window=window_size)
        if self.model is None:
            self.load_model(version=version)
        if len(data) < window_size:
            raise ValueError("Insufficient data for prediction.")
        if hasattr(self, 'is_transformer') and self.is_transformer:
            last_window = data[self.features].iloc[-window_size:].values.reshape(1, window_size, len(self.features))
        else:
            last_window = data[self.features].iloc[-window_size:].values.flatten().reshape(1, -1)
        predicted_prob = self.model.predict(last_window)
        predicted_action = 1 if predicted_prob[0][0] > 0.5 else 0
        return predicted_action, predicted_prob
    
       
class MarketAdviserReverse:
    def __init__(self, model_dir='analyzers/models'):
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, 'market_adviser_down.h5')
        self.features = [
            'Open', 'High', 'Low', 'Close', 'Hour', 'DayOfWeek',
            'RollingMean_Close', 'RollingStd_Close',
            'RollingMean_High', 'RollingMean_Low',
            'PctChange_Close', 'High_Low_Range'
        ]
        self.model = None
        os.makedirs(self.model_dir, exist_ok=True)

    def preprocess_data(self, data, window=10):
        data['Date'] = pd.to_datetime(data['Date'])
        data['Hour'] = data['Date'].dt.hour
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        data['RollingMean_Close'] = data['Close'].rolling(window=window).mean()
        data['RollingStd_Close'] = data['Close'].rolling(window=window).std()
        data['RollingMean_High'] = data['High'].rolling(window=window).mean()
        data['RollingMean_Low'] = data['Low'].rolling(window=window).mean()
        data['PctChange_Close'] = data['Close'].pct_change()
        data['High_Low_Range'] = data['High'] - data['Low']
        data.dropna(inplace=True)
        return data

    def calculate_profit_label(self, data, window=10):
        """
        Create the target label for predicting a downward trend.
        - 1 if any of the next 5 days has a lower price than the current day.
        - 0 otherwise.
        """
        profit_labels = []
        for i in range(len(data) - window):
            current_price = data['Close'].iloc[i]
            future_prices = data['Close'].iloc[i + 1:i + 6]
            profit_labels.append(1 if (future_prices < current_price).any() else 0)
        profit_labels.extend([None] * window)
        data['Profit_Label'] = profit_labels
        data.dropna(subset=['Profit_Label'], inplace=True)
        data['Profit_Label'] = data['Profit_Label'].astype(int)
        return data

    def build_model(self, input_shape):
        model = Sequential([
            Dense(128, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  
        ])
        model.compile(optimizer=Adam(learning_rate=0.0005),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def build_transformer_model(self, input_shape):
        inputs = Input(shape=input_shape)
        inputs = PositionalEncoding(input_shape[0], input_shape[1])(inputs)
        attention_output = MultiHeadAttention(num_heads=4, key_dim=input_shape[-1])(inputs, inputs)
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
        feed_forward = Dense(128, activation='relu')(attention_output)
        feed_forward = Dropout(0.3)(feed_forward)
        feed_forward = LayerNormalization(epsilon=1e-6)(feed_forward)
        flattened = Flatten()(feed_forward)
        outputs = Dense(1, activation='sigmoid')(flattened)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.0005),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def train_model(self, dataset_path, epochs=100, batch_size=32, transformer=False, window_size=20,
                    version=None, window=10):
        """Train the model using the dataset and save it."""
        data = pd.read_csv(dataset_path)
        data = self.preprocess_data(data, window=window)
        data = self.calculate_profit_label(data, window=window)
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[self.features].iloc[i - window_size:i].values.flatten())
            y.append(data['Profit_Label'].iloc[i])
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        sequence_length = window_size
        feature_size = len(self.features)
        usable_samples = len(X) - (len(X) % sequence_length)
        if transformer:
            X = X[:usable_samples] 
            y = y[:usable_samples]
            X = np.array(X)
            y = np.array(y)
            if len(X) > 0:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_train = X_train.reshape((-1, sequence_length, feature_size))
            else:
                raise ValueError("Not enough samples to perform train-test split.")
            X_train = X_train.reshape((-1, sequence_length, feature_size))
            X_test = X_test.reshape((-1, sequence_length, feature_size))
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       
        if self.model is None:
            input_shape = (sequence_length, feature_size) if transformer else (X_train.shape[1],)
            if transformer:
                self.model = self.build_transformer_model(input_shape)
            else:
                self.model = self.build_model(input_shape)
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=max(1, int(epochs * 0.1)),
            restore_best_weights=True,
            min_delta=0.05,
            mode='max'
        )
        
        _ = self.model.fit(
            X_train, y_train, epochs=epochs, batch_size=batch_size,
            validation_data=(X_test, y_test), callbacks=[early_stopping]
        )
        if version is not None:
            self.model.save(f"analyzers/models\market_adviser_down_{version}.h5")
            print(f"analyzers/models\market_adviser_down_{version}.h5")
        else:
            self.model.save(self.model_path)
            print(f"Model trained and saved at {self.model_path}")
        self.is_transformer = transformer
        transformer_flag_path = os.path.join(self.model_dir, 'is_transformer.txt')
        with open(transformer_flag_path, 'w') as f:
            f.write(str(self.is_transformer))
        final_acc = self.model.evaluate(X_test, y_test, verbose=0)[1]
        print(f"Final Test Accuracy: {final_acc:.2f}")

    def load_model(self, version=None):
        """Load the trained model."""
        if os.path.exists(self.model_path):
            if version is not None:
                self.model = tf.keras.models.load_model(f"analyzers/models\market_adviser_down_{version}.h5")
                print(f"Model loaded from analyzers/models\market_adviser_down_{version}.h5")
            else:
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Model loaded from {self.model_path}")
            transformer_flag_path = os.path.join(self.model_dir, 'is_transformer.txt')
            if os.path.exists(transformer_flag_path):
                with open(transformer_flag_path, 'r') as f:
                    self.is_transformer = f.read() == 'True'
            else:
                self.is_transformer = False
            print("Model loaded.")
        else:
            raise FileNotFoundError("Model not found. Train the model first.")

    def predict_last_recommended_action(self, dataset_path, window_size=20, version=None):
        """Predict the recommended action for the last data point."""
        data = pd.read_csv(dataset_path)
        data = self.preprocess_data(data, window=window_size)
        if self.model is None:
            self.load_model(version=version)
        if len(data) < window_size:
            raise ValueError("Insufficient data for prediction.")
        if hasattr(self, 'is_transformer') and self.is_transformer:
            last_window = data[self.features].iloc[-window_size:].values.reshape(1, window_size, len(self.features))
        else:
            last_window = data[self.features].iloc[-window_size:].values.flatten().reshape(1, -1)
        predicted_prob = self.model.predict(last_window)
        predicted_action = 1 if predicted_prob[0][0] > 0.5 else 0
        return predicted_action, predicted_prob
    