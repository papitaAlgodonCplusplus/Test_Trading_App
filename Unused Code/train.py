import threading
import argparse
from training_service import train_model
from dash_app import app as dash_app
from data_service import get_stock_prices, get_dates, get_key_levels, get_times
import webbrowser
import traceback

def start(using_dash_app=True, model_name='DDQN', stock_name='^USD-EUR', window_size=30, num_episode=1, initial_balance=50000):
    stock_prices = get_stock_prices(stock_name)
    dates = get_dates(stock_name)
    #times = get_times(stock_name)
    key_levels = get_key_levels(stock_name)

    # Fusion date and time
    #dates = [f"{date} {time}" for date, time in zip(dates, times)]

    if using_dash_app:
        # Start the Dash app in a separate thread
        threading.Thread(target=lambda: dash_app.run_server(debug=False), daemon=True).start()
        # Open the Dash app in the default web browser
        webbrowser.open("http://127.0.0.1:8050", new=2)
        
    # Run training (this will push data to the queue)
    print("Start training with: ", model_name, stock_name, window_size, num_episode, initial_balance)

    try:
        returns_across_episodes = train_model(using_dash_app,
            model_name, stock_name, window_size, num_episode, initial_balance, stock_prices, dates, key_levels
        )
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
