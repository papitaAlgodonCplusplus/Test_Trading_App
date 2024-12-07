import time
import threading
import traceback
from datetime import datetime
from data_helpers.plotter import Plotter
from data_helpers.data_storage import DataStorage
from strategies.fearless_strategy import main_logic

def schedule_main(file_path, plotter, storage):
    """Run the main function every 60 seconds."""
    while True:
        try:
            risk_ratio = 1/1
            initial_capital=50000 * risk_ratio
            main_logic(file_path, plotter, storage, context_window=None, expected_profit=None, initial_capital=initial_capital, u_a=False, deep_analysis=False, reverse=False,
            bot_type=1, rt=False)
            print(f"Updated at {datetime.now()}")
        except Exception as e:
            print(f"Error occurred: {e}")
            traceback.print_exc()
        time.sleep(0.5)

if __name__ == "__main__":
    try:
        real_time_csv = "data/real_time_data.csv"
        plotter = Plotter()
        storage = DataStorage()
        plotter.start()

        scheduler_thread = threading.Thread(target=schedule_main, args=(real_time_csv, plotter, storage))
        scheduler_thread.daemon = True
        scheduler_thread.start()

        while True:
            time.sleep(0.5)

    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()