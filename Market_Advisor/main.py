
import time
import threading
import traceback
from datetime import datetime
from data_helpers.plotter import Plotter
from strategies.fearless_strategy import main_logic

def schedule_main(file_path, plotter):
    """Run the main function every 60 seconds."""
    while True:
        try:
            main_logic(file_path, plotter, context_window=20, expected_profit=200, initial_capital=50000)
            print(f"Updated at {datetime.now()}")
        except Exception as e:
            print(f"Error occurred: {e}")
        time.sleep(60)

if __name__ == "__main__":
    try:
        real_time_csv = "data/real_time_data.csv"
        plotter = Plotter()
        plotter.start()  # Start the Dash app in a separate thread

        # Run the main function in a loop
        scheduler_thread = threading.Thread(target=schedule_main, args=(real_time_csv, plotter))
        scheduler_thread.daemon = True
        scheduler_thread.start()

        # Keep the main thread alive
        while True:
            time.sleep(50)

    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()