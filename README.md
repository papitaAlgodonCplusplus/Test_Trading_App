# Deep Reinforcement Stock Trading

This project demonstrates a deep reinforcement learning model for stock trading, along with a real-time dashboard to visualize trading performance.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/papitaAlgodonCplusplus/Test_Trading_App.git
    cd Deep-Reinforcement-Stock-Trading
    ```

2. Create a virtual environment (optional but recommended):

    ```sh
    python -m venv venv
    venv\Scripts\activate
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Running the Application

1. Start the training and the Dash app:

    ```sh
    python main.py
    ```

2. Open your web browser and go to `http://127.0.0.1:8050` to view the real-time trading dashboard.

## Command Line Options

You can customize the training parameters using command line options:

- `--dash_app`: Using the dash_app to visualize backtesting (default: `true`)
- `--model_name`: Name of the model (default: `DDQN`)
- `--stock_name`: Stock name (default: `^GSPC_2010-2015`)
- `--window_size`: Span (days) of observation (default: `30`)
- `--num_episode`: Number of episodes (default: `1`)
- `--initial_balance`: Initial balance (default: `50000`)

Example:

```sh
python main.py --model_name DQN --stock_name AAPL_2015-2020 --window_size 50 --num_episode 10 --initial_balance 100000
```
## Shutting Down
To gracefully shut down the application, press `Ctrl+C` in the terminal where the app is running.

## Troubleshooting
If you encounter any issues, please check the following:

1. Ensure all dependencies are installed correctly.
2. Verify that you are using the correct version of Python.
3. Check for any error messages in the terminal and address them accordingly.

For further assistance, please open an issue on the GitHub repository.