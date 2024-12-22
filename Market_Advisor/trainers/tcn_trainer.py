import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analyzers.tcn import TemporalConvolutionalNetwork
if __name__ == '__main__':
    tcn = TemporalConvolutionalNetwork(data_path="data/real_time_data.csv", window_size=70, epochs=4)
    tcn.run(version="1h")
