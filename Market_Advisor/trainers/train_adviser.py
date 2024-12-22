import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from analyzers.market_adviser import MarketAdviserReverse, MarketAdviser
import os
import traceback

def main():
    adviser = MarketAdviser()
    reverse_adviser = MarketAdviserReverse()
    dataset_path = 'data/real_time_data.csv'

    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please provide a valid dataset.")
        return
    try:
        print("Starting model training...")
        adviser.train_model(dataset_path, epochs=20, batch_size=32, transformer=True, version="1h", window=90)
        reverse_adviser.train_model(dataset_path, epochs=20, batch_size=32, transformer=True, version="1h", window=90)
        print("Model training complete.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
