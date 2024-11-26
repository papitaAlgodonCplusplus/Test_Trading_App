import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Model persistence
MODEL_PATH = "market_analysis_model.pkl"

# Initialize the model
market_model = None

def train_market_model(features, labels):
    """
    Train the machine learning model using provided features and labels.

    Args:
        features: 2D array of features.
        labels: 1D array of labels.

    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Model trained with accuracy: {accuracy_score(y_test, predictions):.2f}")
    joblib.dump(model, MODEL_PATH)  # Save the trained model
    return model

def load_or_train_market_model(features=None, labels=None):
    """
    Load an existing model or train a new one if no model exists.

    Args:
        features: Optional 2D array of features for training.
        labels: Optional 1D array of labels for training.

    Returns:
        RandomForestClassifier: Loaded or trained model.
    """
    global market_model
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        market_model = joblib.load(MODEL_PATH)
    elif features is not None and labels is not None:
        print("Training new model...")
        market_model = train_market_model(features, labels)
    else:
        raise ValueError("No model found and no training data provided.")
    return market_model

# Example feature engineering function
def extract_features(balance, risk, stock_prices, t, inventory, window_size):
    """
    Generate features for the ML model based on market data and current state.

    Args:
        balance: Current balance of the agent.
        risk: Risk level of the agent.
        stock_prices: List of historical stock prices.
        t: Current time step.
        inventory: Current inventory held by the agent.
        window_size: Size of the sliding window for price data.

    Returns:
        np.array: Array of features for the ML model.
    """
    # Example features
    recent_prices = stock_prices[max(0, t - window_size):t]
    price_change = np.diff(recent_prices) if len(recent_prices) > 1 else [0]
    avg_price = np.mean(recent_prices) if len(recent_prices) > 0 else 0
    inventory_value = sum(inventory) * (recent_prices[-1] if len(recent_prices) > 0 else 0)

    # Example auto-training setup
    # Normally, you'd collect historical labeled data instead of dummy labels like here
    features = np.array([
        balance,
        risk,
        avg_price,
        inventory_value,
        price_change[-1] if len(price_change) > 0 else 0,
        len(inventory),
    ])

    # Auto-train the model if features and labels are available
    global market_model
    if market_model is None:
        print("Auto-training model...")
        # Example dummy data for training (replace this with real historical data)
        dummy_features = np.random.rand(100, len(features))
        dummy_labels = np.random.choice([0, 1, 2], 100)  # 0: Hold, 1: Buy, 2: Sell
        load_or_train_market_model(dummy_features, dummy_labels)

    return features

def obtain_forbidden_actions(balance, risk, stock_prices, t, inventory, action_dict, window_size):
    """
    Determine forbidden and recommended actions using market intelligence.

    Args:
        balance: Current balance of the agent.
        risk: Risk level of the agent.
        stock_prices: List of historical stock prices.
        t: Current time step.
        inventory: Current inventory held by the agent.
        action_dict: Dictionary mapping action indices to action names.
        window_size: Size of the sliding window for price data.

    Returns:
        tuple: (forbidden_actions, recommended_actions)
    """
    # Extract features for the current market state
    features = extract_features(balance, risk, stock_prices, t, inventory, window_size)
    features = features.reshape(1, -1)  # Reshape for model input

    # Predict probabilities for each action (0: Hold, 1: Buy, 2: Sell)
    probabilities = market_model.predict_proba(features)[0]

    # Define thresholds for recommending or forbidding actions
    forbidden_threshold = 0.2  # Actions with probabilities < 0.2 are forbidden
    recommended_threshold = 0.6  # Actions with probabilities > 0.6 are recommended

    # Determine forbidden and recommended actions
    forbidden_actions = [action for action, prob in enumerate(probabilities) if prob < forbidden_threshold]
    recommended_actions = [action for action, prob in enumerate(probabilities) if prob > recommended_threshold]

    return forbidden_actions, recommended_actions
