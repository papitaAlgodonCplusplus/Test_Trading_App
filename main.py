import time
import argparse
from train import start

def main(using_dash_app, model_name, stock_name, window_size, num_episode, initial_balance):
    try:
        print(f"Model: {model_name}")
        print(f"Stock: {stock_name}")
        print(f"Window Size: {window_size} days")
        print(f"Number of Episodes: {num_episode}")
        print(f"Initial Balance: ${initial_balance}")

        if using_dash_app:
            print("Starting training and Dash app...")
            start(
                using_dash_app=True,
                model_name=model_name,
                stock_name=stock_name,
                window_size=window_size,
                num_episode=num_episode,
                initial_balance=initial_balance,
            )
        else:
            print("Starting training without Dash app...")
            start(
                using_dash_app=False,
                model_name=model_name,
                stock_name=stock_name,
                window_size=window_size,
                num_episode=num_episode,
                initial_balance=initial_balance,
            )

        # Keep the main script alive to monitor the processes
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line options for training.')
    
    # Dash app control
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dash_app', action='store_true', dest='using_dash_app', help='Start Dash app')
    group.add_argument('--no_dash_app', action='store_false', dest='using_dash_app', help='Do not start Dash app')

    # Training-specific options
    parser.add_argument('--model_name', action="store", dest="model_name", default='DDQN', help="model name")
    parser.add_argument('--stock_name', action="store", dest="stock_name", default='^GSPC_2010-2015', help="stock name")
    parser.add_argument('--window_size', action="store", dest="window_size", default=30, type=int, help="span (days) of observation")
    parser.add_argument('--num_episode', action="store", dest="num_episode", default=1, type=int, help='episode number')
    parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int, help='initial balance')

    # Default Dash app behavior
    parser.set_defaults(using_dash_app=True)

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(
        args.using_dash_app,
        args.model_name,
        args.stock_name,
        args.window_size,
        args.num_episode,
        args.initial_balance,
    )