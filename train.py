import threading
import argparse
from training_service import train_model
from data_service import get_stock_prices, get_dates
from dash_app import app as dash_app

def start():
    parser = argparse.ArgumentParser(description='command line options')
    parser.add_argument('--model_name', action="store", dest="model_name", default='DDQN', help="model name")
    parser.add_argument('--stock_name', action="store", dest="stock_name", default='^GSPC_2010-2015', help="stock name")
    parser.add_argument('--window_size', action="store", dest="window_size", default=30, type=int, help="span (days) of observation")
    parser.add_argument('--num_episode', action="store", dest="num_episode", default=1, type=int, help='episode number')
    parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int, help='initial balance')
    inputs = parser.parse_args()

    model_name = inputs.model_name
    stock_name = inputs.stock_name
    window_size = inputs.window_size
    num_episode = inputs.num_episode
    initial_balance = inputs.initial_balance

    stock_prices = get_stock_prices(stock_name)
    dates = get_dates(stock_name)

    # Start the Dash app in a separate thread
    threading.Thread(target=lambda: dash_app.run_server(debug=False), daemon=True).start()

    # Run training (this will push data to the queue)
    returns_across_episodes = train_model(model_name, stock_name, window_size, num_episode, initial_balance, stock_prices, dates)
