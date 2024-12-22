import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from analyzers.market_adviser import MarketAdviser, MarketAdviserReverse

def indicate(data_path="data/real_time_data.csv", version=None):
    adviser = MarketAdviser()
    reverse_adviser = MarketAdviserReverse()
    _, predicted_prob = adviser.predict_last_recommended_action(data_path, version=version)
    _, reversed_predicted_prob = reverse_adviser.predict_last_recommended_action(data_path, version=version)
    print(f"Predicted Prob: {predicted_prob}, Reversed Predicted Prob: {reversed_predicted_prob}")
    if predicted_prob[[0]] > reversed_predicted_prob[[0]] and predicted_prob[[0]] > 0.9:
        return 1
    else:
        return -1
