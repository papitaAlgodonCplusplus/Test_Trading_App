import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from analyzers.tcn import TemporalConvolutionalNetwork

def indicate(data_path="data/real_time_data.csv", version=None):
    tcn = TemporalConvolutionalNetwork(data_path="data/real_time_data.csv")
    prediction, probs = tcn.predict(data_path, version)
    print(f"Prediction: {prediction}, Probabilities: {probs}")
   
    if prediction == 1 or prediction == -1:
        return prediction
    else:
        return 0