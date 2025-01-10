import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from analyzers.tcn import TemporalConvolutionalNetwork

def indicate(data_path="data/real_time_data.csv", version=None):
    return 0