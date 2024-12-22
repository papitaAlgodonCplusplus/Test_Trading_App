import sys
import threading
import time
import traceback
import pandas as pd
import csv
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton,
    QLabel, QSpinBox, QLineEdit, QComboBox, QFormLayout, QMessageBox
)
from PyQt5.QtCore import QRunnable, QThreadPool
from data_helpers.plotter import Plotter
from data_helpers.data_storage import DataStorage
from strategies.fearless_strategy import main_logic
import MetaTrader5 as mt5
import datetime

def clear_csv(file_path):
    """Clear the content of a CSV file."""
    if os.path.exists(file_path):
        with open(file_path, 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([])  # Clear content


def schedule_main(file_path, plotter, storage, params, log_file):
    """Run the main function periodically with given parameters."""
    try:
        while True:
            try:
                main_logic(
                    file_path, plotter, storage, 
                    context_window=200, expected_profit=None,
                    initial_capital=params["initial_capital"], 
                    u_a=False, deep_analysis=False, reverse=False,
                    bot_type=params["bot_type"], rt=True, freeze=13,
                    forced_patience=True, stop_loss_pips=params["stop_loss_pips"], 
                    take_profit_pips=params["take_profit_pips"], pip_value=0.0001, 
                    target_rr_ratio=params["take_profit_pips"] / params["stop_loss_pips"],
                    code_multiplier=params["code_multiplier"], 
                    log_file=log_file, trailing_stop_loss=True,
                    advisors_threshold=2, web_scraping=params["web_scraping"]
                )
            except Exception as e:
                print(f"Error occurred in {log_file}: {e}")
                traceback.print_exc()
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("Exiting... Clearing CSV data.")
        clear_csv(file_path)


def start_bot(port, stop_loss, take_profit, multiplier, initial_capital, log_file, web_scraping):
    """Start a bot with specific configuration."""
    real_time_csv = "data/real_time_data.csv"
    plotter = Plotter()
    storage = DataStorage()
    plotter.start(port)

    if not os.path.exists(log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'w') as f:
            f.write("Log initialized\n")

    params = {
        "initial_capital": initial_capital,
        "bot_type": 7,
        "stop_loss_pips": stop_loss,
        "take_profit_pips": take_profit,
        "code_multiplier": multiplier,
        "web_scraping": web_scraping
    }

    thread = threading.Thread(
        target=schedule_main, 
        args=(real_time_csv, plotter, storage, params, log_file)
    )
    thread.daemon = True
    thread.start()


class BotRunnable(QRunnable):
    def __init__(self, port, stop_loss, take_profit, multiplier, initial_capital, log_file, web_scraping):
        super().__init__()
        self.port = port
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.multiplier = multiplier
        self.initial_capital = initial_capital
        self.log_file = log_file
        self.web_scraping = web_scraping

    def run(self):
        try:
            start_bot(
                port=self.port,
                stop_loss=self.stop_loss,
                take_profit=self.take_profit,
                multiplier=self.multiplier,
                initial_capital=self.initial_capital,
                log_file=self.log_file,
                web_scraping=self.web_scraping
            )
        except Exception as e:
            print(f"Error starting bot: {e}")


class BotUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bot Management")
        self.setGeometry(100, 100, 600, 400)
        self.thread_pool = QThreadPool()
        self.used_ports = set()
        self.used_multipliers = set()

        # Main layout
        self.main_layout = QVBoxLayout()

        # Form layout for bot parameters
        self.form_layout = QFormLayout()
        self.port_input = QSpinBox()
        self.port_input.setRange(1000, 9999)
        self.stop_loss_input = QSpinBox()
        self.stop_loss_input.setRange(1, 100)
        self.take_profit_input = QSpinBox()
        self.take_profit_input.setRange(1, 100)
        self.multiplier_input = QComboBox()
        self.multiplier_input.addItems(["1", "2", "3", "4", "5"])
        self.log_file_input = QLineEdit("temp_files/log.txt")
        self.web_scraping_input = QComboBox()
        self.web_scraping_input.addItems(["5m", "15m", "1h"])
        self.initial_capital_input = QSpinBox()
        self.initial_capital_input.setRange(1000, 1000000)
        self.initial_capital_input.setValue(50000)

        # Add fields to form
        self.form_layout.addRow("Port:", self.port_input)
        self.form_layout.addRow("Stop Loss (pips):", self.stop_loss_input)
        self.form_layout.addRow("Take Profit (pips):", self.take_profit_input)
        self.form_layout.addRow("Multiplier:", self.multiplier_input)
        self.form_layout.addRow("Log File:", self.log_file_input)
        self.form_layout.addRow("Web Scraping Interval:", self.web_scraping_input)
        self.form_layout.addRow("Initial Capital:", self.initial_capital_input)

        # Add Start button
        self.start_button = QPushButton("Start Bot")
        self.start_button.clicked.connect(self.start_bot)

        # Status label
        self.status_label = QLabel("Status: Idle")

        # Add to main layout
        self.main_layout.addLayout(self.form_layout)
        self.main_layout.addWidget(self.start_button)
        self.main_layout.addWidget(self.status_label)

        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

    def start_bot(self):
        # Get parameters
        port = self.port_input.value()
        stop_loss = self.stop_loss_input.value()
        take_profit = self.take_profit_input.value()
        multiplier = int(self.multiplier_input.currentText())
        log_file = self.log_file_input.text()
        web_scraping = self.web_scraping_input.currentText()
        initial_capital = self.initial_capital_input.value()

        # Check if port is already in use
        if port in self.used_ports:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Error")
            msg_box.setText(f"Port {port} is already in use. Please choose a different port.")
            msg_box.exec_()
            return

        # Check if multiplier is already used
        if multiplier in self.used_multipliers:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Error")
            msg_box.setText(f"Multiplier {multiplier} is already in use. Please choose a different multiplier.")
            msg_box.exec_()
            return

        self.used_ports.add(port)
        self.used_multipliers.add(multiplier)

        # Start bot in a separate thread
        bot_runnable = BotRunnable(port, stop_loss, take_profit, multiplier, initial_capital, log_file, web_scraping)
        self.thread_pool.start(bot_runnable)

        # Display success message
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Bot Created")
        msg_box.setText(f"Bot has been successfully created!\nYou can access it at: http://localhost:{port}")
        msg_box.addButton("Continue with a new bot", QMessageBox.AcceptRole)
        msg_box.exec_()

        self.status_label.setText(f"Bot started on port {port}")

    def closeEvent(self, event):
        self.thread_pool.clear()  # Clear all threads before closing
        event.accept()


def write_csv(output_file):
    """Write data row by row from input_file to output_file."""
    # Initialize the MT5 terminal
    if not mt5.initialize():
        print("Initialize() failed, error code =", mt5.last_error())
        quit()

    # Define symbol and timeframe
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_M1  # Minute-by-minute timeframe

    # Initialize the CSV file if it doesn't exist
    if not os.path.exists(output_file):
        # Create an empty DataFrame with the necessary columns
        initial_df = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        initial_df.to_csv(output_file, index=False)

    try:
        while True:
            # Load existing data to check the latest timestamp
            existing_data = pd.read_csv(output_file)
            
            # Get the latest timestamp from the CSV, if available
            last_timestamp = None
            if not existing_data.empty:
                last_timestamp = pd.to_datetime(existing_data['Date'].iloc[-1])

            # Fetch the latest minute data
            now = datetime.datetime.now() + datetime.timedelta(hours=2)
            rates = mt5.copy_rates_from(symbol, timeframe, now, 1)

            if rates is not None and len(rates) > 0:
                new_data = pd.DataFrame(rates)

                # Format the data to match the desired structure
                new_data['Date'] = pd.to_datetime(new_data['time'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
                new_data = new_data.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'tick_volume': 'Volume'
                })
                new_data = new_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

                # Check if the data is new based on the timestamp
                if last_timestamp is None or pd.to_datetime(new_data['Date'].iloc[0]) > last_timestamp:
                    # Append new data to the CSV
                    new_data.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
                    print(f"New data added at {new_data['Date'].iloc[0]}")
                else:
                    print("No new data available.")
            else:
                print("No data retrieved from MetaTrader 5.")

            # Wait for 60 seconds before fetching new data again
            time.sleep(60)

    except KeyboardInterrupt:
        print("Data collection stopped by user.")

    finally:
        # Shutdown MT5 connection
        mt5.shutdown()


# Start CSV Writer Thread
def start_csv_writer(output_file):
    threading.Thread(target=write_csv, args=(output_file,), daemon=True).start()

# Entry Point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    output_file = "data/real_time_data.csv"
    start_csv_writer(output_file)

    # Launch UI
    window = BotUI()
    window.show()
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("Exiting application and clearing CSV file.")
        clear_csv(output_file)
