import MetaTrader5 as mt5
import pandas as pd
import datetime
import time
import os

# Initialize the MT5 terminal
if not mt5.initialize():
    print("Initialize() failed, error code =", mt5.last_error())
    quit()

# Define symbol and timeframe
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M1  # Minute-by-minute timeframe
csv_file = 'data/real_time_data.csv'

# Initialize the CSV file if it doesn't exist
if not os.path.exists(csv_file):
    # Create an empty DataFrame with the necessary columns
    initial_df = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    initial_df.to_csv(csv_file, index=False)

try:
    while True:
        # Load existing data to check the latest timestamp
        existing_data = pd.read_csv(csv_file)
        
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
                new_data.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
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
