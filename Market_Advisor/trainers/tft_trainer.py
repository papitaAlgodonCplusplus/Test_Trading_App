import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analyzers.tft import TemporalFusionTransformer
if __name__ == '__main__':
    tft = TemporalFusionTransformer(data_path="data/real_time_data.csv", window_size=70, epochs=8)
    tft.run(version="1h")