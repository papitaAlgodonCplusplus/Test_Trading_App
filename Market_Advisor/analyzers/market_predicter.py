import pandas as pd
import numpy as np
import ta
import os
import argparse
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=["Date"])
    return data

def statistical_indicators(data):
    # Statistical Measures
    data["Z_Score_Close"] = (data["Close"] - data["Close"].rolling(window=20).mean()) / data["Close"].rolling(window=20).std()
    data["Skewness"] = data["Close"].rolling(window=20).skew()
    data["Kurtosis"] = data["Close"].rolling(window=20).kurt()
    data.fillna(0, inplace=True)
    return data

def candlestick_patterns(data):
    data["Hammer"] = ((data["Close"] > data["Open"]) & ((data["Close"] - data["Low"]) > 2 * (data["Open"] - data["Close"]))).astype(int)
    data["Shooting_Star"] = ((data["Open"] > data["Close"]) & ((data["High"] - data["Open"]) > 2 * (data["Open"] - data["Close"]))).astype(int)
    data["Doji"] = ((abs(data["Close"] - data["Open"]) / (data["High"] - data["Low"])) < 0.1).astype(int)
    data["Bullish_Engulfing"] = ((data["Close"] > data["Open"]) & 
                                 (data["Close"].shift(1) < data["Open"].shift(1)) & 
                                 (data["Close"] > data["Open"].shift(1)) & 
                                 (data["Open"] < data["Close"].shift(1))).astype(int)
    data["Bearish_Engulfing"] = ((data["Close"] < data["Open"]) & 
                                  (data["Close"].shift(1) > data["Open"].shift(1)) & 
                                  (data["Open"] > data["Close"].shift(1)) & 
                                  (data["Close"] < data["Open"].shift(1))).astype(int)
    data.fillna(0, inplace=True)
    return data

def fibonacci_retracement(data):
    data["Fib_High"] = data["High"].cummax()
    data["Fib_Low"] = data["Low"].cummin()
    data["Fib_Retrace"] = (data["Close"] - data["Fib_Low"]) / (data["Fib_High"] - data["Fib_Low"])
    data["Support"] = data["Low"].rolling(window=10).min()
    data["Resistance"] = data["High"].rolling(window=10).max()
    data.fillna(0, inplace=True)
    return data

def moving_averages(data):
    data["MA_20"] = data["Close"].rolling(window=20).mean()
    data["MA_50"] = data["Close"].rolling(window=50).mean()
    data["MA_200"] = data["Close"].rolling(window=200).mean()
    data.fillna(0, inplace=True)
    return data

def trend(data):
    data["ATR"] = ta.volatility.AverageTrueRange(data["High"], data["Low"], data["Close"], window=14).average_true_range()
    data["Volatility"] = ta.volatility.BollingerBands(data["Close"]).bollinger_hband() - ta.volatility.BollingerBands(data["Close"]).bollinger_lband()
    data["Trendline"] = data["Close"].rolling(window=20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    data["Mean_Reversion"] = data["Close"] - data["Close"].rolling(window=20).mean()
    data.fillna(0, inplace=True)
    return data

def momentum(data):
    # Core Momentum Indicators
    data["RSI"] = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()
    data["Momentum"] = data["Close"].pct_change(periods=10)
    data["ROC"] = ta.momentum.ROCIndicator(data["Close"], window=12).roc()
    macd = ta.trend.MACD(data["Close"])
    data["MACD"] = macd.macd()
    data["Stoch_Oscillator"] = ta.momentum.StochasticOscillator(data["High"], data["Low"], data["Close"]).stoch()
    data.fillna(0, inplace=True)
    return data

def preprocess_data(data):
    data.fillna(0, inplace=True)
    return data

def prepare_lstm_data(data, lookback=10):
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    features = data.drop(["Date", "Target"], axis=1).values
    target = data["Target"].values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i-lookback:i])
        y.append(target[i])
    
    flat_features = features[lookback:]
    return np.array(X), flat_features, np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_mlp_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=2, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Conv1D(64, kernel_size=2, activation='relu'),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_lstm(X_train, y_train, X_val, y_val, batch_size=64, epochs=50):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(
        monitor="val_accuracy",  
        min_delta=0.01,          
        patience=30,            
        verbose=1,              
        restore_best_weights=True  
    )
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stopping]  
    )
    return model

def train_mlp(X_train, y_train, X_val, y_val, input_dim, epochs=50, batch_size=32):
    model = build_mlp_model(input_dim)
    
    
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        min_delta=0.01,
        patience=30,
        verbose=1,
        restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stopping]
    )
    return model

def train_cnn(X_train, y_train, X_val, y_val, input_shape, epochs=50, batch_size=32):
    model = build_cnn_model(input_shape)
    
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        min_delta=0.01,
        patience=30,
        verbose=1,
        restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stopping]
    )
    return model

def stack_models(lstm_predictions, mlp_predictions, cnn_predictions, y_val):
    stacked_features = np.vstack((lstm_predictions, mlp_predictions, cnn_predictions)).T
    meta_model = Sequential([
        Dense(16, activation='relu', input_dim=stacked_features.shape[1]),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    meta_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        min_delta=0.01,
        patience=20,
        verbose=1,
        restore_best_weights=True
    )

    meta_model.fit(stacked_features, y_val, epochs=50, batch_size=16, verbose=1, callbacks=[early_stopping])
    return meta_model

def plot_predictions(data, predictions):
    plt.figure(figsize=(14, 7))
    plt.plot(data["Date"], data["Close"], label="Close Prices", color="blue", linewidth=2)
    for date, close, prediction in zip(data["Date"], data["Close"], predictions):
        color = "green" if prediction == 1 else "red"
        plt.scatter(date, close, color=color, edgecolor="black", s=50, label="Prediction")
    plt.title("Close Prices with Model Predictions", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Close Price", fontsize=12)
    plt.legend(["Close Prices", "Predictions"], loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False,
                        help="Load pre-trained models instead of training new ones")
    args = parser.parse_args()
    data = load_data("data.csv")

    functions_to_test = [preprocess_data]
    # , statistical_indicators, candlestick_patterns, fibonacci_retracement, moving_averages, trend, momentum
    for func in functions_to_test:
        data = func(data)
        X_lstm, X_flat, y = prepare_lstm_data(data)
        X_train_lstm, X_val_lstm, X_train_flat, X_val_flat, y_train, y_val = train_test_split(
            X_lstm, X_flat, y, test_size=0.2, random_state=42
        )

        if args.load_model:
            lstm_model = load_model("./analyzers/models/lstm_model.h5")
            mlp_model = load_model("./analyzers/models/mlp_model.h5")
            cnn_model = load_model("./analyzers/models/cnn_model.h5")
            meta_model = load_model("./analyzers/models/meta_model.h5")
            lstm_predictions = (lstm_model.predict(X_val_lstm).flatten() > 0.5).astype(int)
            mlp_predictions = (mlp_model.predict(X_val_flat).flatten() > 0.5).astype(int)
            cnn_predictions = (cnn_model.predict(X_val_flat.reshape(X_val_flat.shape[0], X_val_flat.shape[1], 1)).flatten() > 0.5).astype(int) 

        else:
            lstm_model = train_lstm(X_train_lstm, y_train, X_val_lstm, y_val)
            mlp_model = train_mlp(X_train_flat, y_train, X_val_flat, y_val, input_dim=X_train_flat.shape[1])
            cnn_model = train_cnn(X_train_flat.reshape(X_train_flat.shape[0], X_train_flat.shape[1], 1), y_train,
                                  X_val_flat.reshape(X_val_flat.shape[0], X_val_flat.shape[1], 1), y_val,
                                  input_shape=(X_train_flat.shape[1], 1))
            lstm_predictions = (lstm_model.predict(X_val_lstm).flatten() > 0.5).astype(int)
            mlp_predictions = (mlp_model.predict(X_val_flat).flatten() > 0.5).astype(int)
            cnn_predictions = (cnn_model.predict(X_val_flat.reshape(X_val_flat.shape[0], X_val_flat.shape[1], 1)).flatten() > 0.5).astype(int)
            meta_model = stack_models(lstm_predictions, mlp_predictions, cnn_predictions, y_val)
            os.makedirs("./analyzers/models/", exist_ok=True)
            lstm_model.save("./analyzers/models/lstm_model.h5")
            mlp_model.save("./analyzers/models/mlp_model.h5")
            cnn_model.save("./analyzers/models/cnn_model.h5")
            meta_model.save("./analyzers/models/meta_model.h5")
            print("Models saved successfully.")
          
            stacked_features = np.vstack((lstm_predictions, mlp_predictions, cnn_predictions)).T
            final_predictions = (meta_model.predict(stacked_features).flatten() > 0.5).astype(int)
            accuracy = accuracy_score(y_val, final_predictions)
            print("Function: ", func.__name__)
            print(f"Stacked Model Accuracy: {accuracy:.4f}")
            plot_predictions(data[-len(y_val):], final_predictions)
            
if __name__ == "__main__":
    main()

def obtain_final_predictions(csv_file):
    import os
    import tensorflow as tf
    import absl.logging
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    absl.logging.set_verbosity(absl.logging.ERROR)
    tf.get_logger().setLevel('ERROR')

    data = load_data(csv_file)
    data = preprocess_data(data)

    lstm_model = load_model("./analyzers/models/lstm_model.h5")
    mlp_model = load_model("./analyzers/models/mlp_model.h5")
    cnn_model = load_model("./analyzers/models/cnn_model.h5")
    meta_model = load_model("./analyzers/models/meta_model.h5")
    X, X_f, _ = prepare_lstm_data(data)
    lstm_predictions = (lstm_model.predict(X).flatten() > 0.5).astype(int)
    mlp_predictions = (mlp_model.predict(X_f).flatten() > 0.5).astype(int)
    cnn_predictions = (cnn_model.predict(X_f.reshape(X_f.shape[0], X_f.shape[1], 1)).flatten() > 0.5).astype(int)
    
    stacked_features = np.vstack((lstm_predictions, mlp_predictions, cnn_predictions)).T
    final_predictions = (meta_model.predict(stacked_features).flatten() > 0.5).astype(int)
    
    return final_predictions, meta_model.predict(stacked_features).flatten()