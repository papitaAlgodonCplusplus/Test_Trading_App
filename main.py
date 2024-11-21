import time
from train import start

if __name__ == "__main__":
    try:
        print("Starting training and Dash app...")
        # Start training and Dash app (Dash runs in its own thread via train.py)
        start()

        # Keep the main script alive to monitor the processes
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error: {e}")
