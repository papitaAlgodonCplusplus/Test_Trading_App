import time
import threading
import traceback
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_helpers.plotter import Plotter
from data_helpers.data_storage import DataStorage
from strategies.fearless_strategy import main_logic

def schedule_main(file_path, plotter, storage):
    """Run the main function every 60 seconds."""
    while True:
        try:
            # 7 for downtrades, 8 for uptrades
            risk_ratio = 1/1
            initial_capital=50000 * risk_ratio
            main_logic(file_path, plotter, storage, context_window=200, expected_profit=None, initial_capital=initial_capital, u_a=False, deep_analysis=False, reverse=False,
            bot_type=7, rt=True, freeze=13, forced_patience=True, stop_loss_pips=3, take_profit_pips=9, pip_value=0.0001, target_rr_ratio=3, code_multiplier=2,
            log_file="temp_files/last_action_log.txt", trailing_stop_loss=True,
            advisors_threshold=2, web_scraping="5m")
        except Exception as e:
            print(f"Error occurred: {e}")
            traceback.print_exc()
        time.sleep(1)

if __name__ == "__main__":
    try:
        real_time_csv = "data/real_time_data.csv"
        plotter = Plotter()
        storage = DataStorage()
        plotter.start(6410)

        scheduler_thread = threading.Thread(target=schedule_main, args=(real_time_csv, plotter, storage))
        scheduler_thread.daemon = True
        scheduler_thread.start()

        while True:
            time.sleep(2)

    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()