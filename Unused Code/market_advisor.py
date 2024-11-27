import pandas as pd
import numpy as np
import ta
import os
import argparse
import matplotlib.pyplot as plt

# Sklearn imports
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# Sci-Keras import
from scikeras.wrappers import KerasClassifier

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=["Date"])
    return data

def add_advanced_technical_indicators(data):
    # Existing technical indicators
    data["RSI"] = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()
    data["Momentum"] = data["Close"].pct_change(periods=10)
    data["ROC"] = ta.momentum.ROCIndicator(data["Close"], window=12).roc()
    
    # Additional advanced indicators
    macd = ta.trend.MACD(data["Close"])
    data["MACD"] = macd.macd()
    data["Signal_Line"] = macd.macd_signal()
    
    # Volume-based indicators
    data["Volume_MA_20"] = data["Volume"].rolling(window=20).mean()
    data["Volume_Oscillator"] = (data["Volume"] - data["Volume_MA_20"]) / data["Volume_MA_20"]
    
    # Advanced price ratio indicators
    data["Price_to_MA_Ratio"] = data["Close"] / data["Close"].rolling(window=20).mean()
    data["Exponential_MA"] = data["Close"].ewm(span=20).mean()
    
    # More complex candlestick patterns
    data["Engulfing_Bullish"] = ((data["Close"] > data["Open"]) & 
                                 (data["Close"].shift(1) < data["Open"].shift(1)) & 
                                 (data["Close"] > data["Open"].shift(1)) & 
                                 (data["Open"] < data["Close"].shift(1))).astype(int)
    
    data["Divergence_RSI"] = ((data["RSI"] > 70) & (data["Close"].pct_change() < 0)).astype(int)
    
    # Remove NaN values
    data.fillna(0, inplace=True)
    return data

def prepare_advanced_data(data, lookback=20):
    # Target: Next day price movement
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    
    # Remove technical indicator columns and Date column
    features = data.drop(["Date", "Close", "Target"], axis=1)
    target = data["Target"]
    
    # Advanced scaling
    print("Scaling features using RobustScaler...")
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)
    print("Features scaled.")
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(lookback, len(features_scaled)):
        X.append(features_scaled[i-lookback:i])
        y.append(target.iloc[i])
    
    return np.array(X), np.array(y), features_scaled

def create_advanced_lstm_model(input_shape, neurons=64, dropout_rate=0.3, learning_rate=0.001):
    model = Sequential([
        LSTM(neurons, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(neurons//2),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def hyperparameter_tuning(X_train_lstm, y_train):
    print("Starting hyperparameter tuning...")
    
    # Get input shape
    input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
    print(f"Input shape: {input_shape}")
    
    def create_model(neurons=64, dropout_rate=0.3, learning_rate=0.001):
        print(f"Creating model with neurons={neurons}, dropout_rate={dropout_rate}, learning_rate={learning_rate}")
        model = Sequential([
            LSTM(neurons, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            LSTM(neurons//2),
            Dropout(dropout_rate),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    # Wrap Keras model for sklearn
    print("Wrapping Keras model for sklearn...")
    model = KerasClassifier(model=create_model, verbose=0)
    
    # Hyperparameter search space
    print("Defining hyperparameter search space...")
    param_dist = {
        'model__neurons': [32, 64, 128],
        'model__dropout_rate': [0.2, 0.3, 0.4],
        'model__learning_rate': [0.001, 0.0005, 0.01],
        'batch_size': [32, 64, 128],
        'epochs': [50, 100, 150]
    }
    
    # Time series cross-validation
    print("Setting up time series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Randomized search
    print("Starting randomized search...")
    random_search = RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_dist, 
        n_iter=10, 
        cv=tscv, 
        scoring='accuracy',
        random_state=42
    )
    
    try:
        print("Fitting randomized search...")
        random_search.fit(X_train_lstm, y_train)
        print("Randomized search completed.")
        print("Best Parameters:", random_search.best_params_)
        
        # Get best parameters
        best_params = random_search.best_params_
        print(f"Best parameters: {best_params}")
        
        # Create and train the best model
        print("Creating and training the best model...")
        best_model = create_model(
            neurons=best_params['model__neurons'],
            dropout_rate=best_params['model__dropout_rate'],
            learning_rate=best_params['model__learning_rate']
        )
        
        # Early stopping
        print("Setting up early stopping...")
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        # Fit the best model
        print("Fitting the best model...")
        best_model.fit(
            X_train_lstm, 
            y_train, 
            epochs=best_params['epochs'], 
            batch_size=best_params['batch_size'], 
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        print("Best model training completed.")
        
        return best_model
    
    except Exception as e:
        print(f"Error in hyperparameter tuning: {e}")
        # Fallback to a default model if tuning fails
        print("Falling back to default model...")
        default_model = create_model()
        default_model.fit(
            X_train_lstm, 
            y_train, 
            epochs=100, 
            batch_size=64, 
            verbose=1
        )
        print("Default model training completed.")
        return default_model

def advanced_ensemble_training(X_train_lstm, y_train, X_train_flat, X_val_lstm, y_val):
    print("Starting advanced ensemble training...")

    # Train multiple base models
    models = []
    
    # LSTM Model with Hyperparameter Tuning
    try:
        print("Training LSTM model with hyperparameter tuning...")
        lstm_model = hyperparameter_tuning(X_train_lstm, y_train)
        models.append(lstm_model)
        print("LSTM model training completed.")
    except Exception as e:
        print(f"LSTM Model Training Failed: {e}")
    
    # Random Forest
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        min_samples_split=5, 
        random_state=42
    )
    rf_model.fit(X_train_flat, y_train)
    models.append(rf_model)
    print("Random Forest model training completed.")
    
    # Gradient Boosting
    print("Training Gradient Boosting model...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42
    )
    gb_model.fit(X_train_flat, y_train)
    models.append(gb_model)
    print("Gradient Boosting model training completed.")
    
    # Prediction function
    def get_oof_predictions(models, X_val_lstm, X_val_flat):
        print("Generating out-of-fold predictions for validation data...")
        oof_predictions = np.column_stack([
            model.predict_proba(X_val_flat)[:, 1] if hasattr(model, 'predict_proba') 
            else model.predict(X_val_lstm) for model in models
        ])
        print("Out-of-fold predictions generated.")
        return oof_predictions
    
    # Validate base models
    print("Validating base models...")
    val_predictions = get_oof_predictions(models, X_val_lstm, X_val_flat)
    print("Base models validation completed.")

    # Meta-Classifier
    print("Training meta-classifier...")
    meta_classifier = LogisticRegression(penalty='l2')
    meta_classifier.fit(val_predictions, y_val)
    print("Meta-classifier training completed.")
    
    print("Advanced ensemble training completed.")
    return models, meta_classifier

def main():
    # Load and preprocess data
    data = load_data("data.csv")
    data = add_advanced_technical_indicators(data)
    
    # Prepare data
    X_lstm, y, X_flat = prepare_advanced_data(data)
    
    # Split data
    X_train_lstm, X_val_lstm, y_train, y_val = train_test_split(
        X_lstm, y, test_size=0.2, shuffle=False
    )
    X_train_flat, X_val_flat = X_flat[:len(X_train_lstm)], X_flat[len(X_train_lstm):]
    
    # Advanced Ensemble Training
    base_models, meta_classifier = advanced_ensemble_training(
        X_train_lstm, y_train, X_train_flat, X_val_lstm, y_val
    )
    
    # Make predictions
    base_val_predictions = np.column_stack([
        model.predict_proba(X_val_flat)[:, 1] if hasattr(model, 'predict_proba') 
        else model.predict(X_val_lstm) for model in base_models
    ])
    
    final_predictions = meta_classifier.predict(base_val_predictions)
    
    # Evaluate
    accuracy = accuracy_score(y_val, final_predictions)
    print(f"Advanced Ensemble Model Accuracy: {accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_val, final_predictions))
    
    # Visualization
    plt.figure(figsize=(15, 7))
    plt.plot(data["Date"][-len(y_val):], data["Close"][-len(y_val):], label="Close Prices")
    plt.scatter(
        data["Date"][-len(y_val):][final_predictions == 1], 
        data["Close"][-len(y_val):][final_predictions == 1], 
        color='green', label='Predicted Increase'
    )
    plt.scatter(
        data["Date"][-len(y_val):][final_predictions == 0], 
        data["Close"][-len(y_val):][final_predictions == 0], 
        color='red', label='Predicted Decrease'
    )
    plt.title("Stock Price Predictions", fontsize=15)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()