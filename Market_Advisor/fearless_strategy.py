import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from market_advisor import obtain_final_predictions

def run_simulation(data, initial_capital=10000):
    """Run a trading simulation based on the 'Action' column and return a profit series."""
    capital = initial_capital
    position = 0  # Number of Euros currently held
    profit_over_time = []

    for i in range(len(data)):
        action = data['Action'].iloc[i]
        price = data['Close'].iloc[i]

        # Execute the action
        if action == 'Buy' and capital > 0:
            position = capital / price  # Buy Euros
            capital = 0
        elif action == 'Sell' and position > 0:
            capital = position * price  # Sell Euros for USD
            position = 0

        # Track total profit/loss
        total_value = capital + (position * price)
        profit_over_time.append(total_value - initial_capital)

    data['Profit'] = profit_over_time

    # Sell all remaining Euros at the end of the simulation
    if position > 0:
        capital = position * data['Close'].iloc[-1]
        position = 0
    return data, capital

def main(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'])

    # Calculate technical indicators
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    rsi = RSIIndicator(data['Close'], window=14)
    data['RSI'] = rsi.rsi()

    # Define Buy/Sell/Hold actions
    data['Action'] = np.where(
        (data['MACD'] > data['Signal_Line']) & (data['RSI'] < 70),  # Buy when MACD > Signal Line and RSI not overbought
        'Buy',
        np.where(
            (data['MACD'] < data['Signal_Line']) & (data['RSI'] > 30),  # Sell when MACD < Signal Line and RSI not oversold
            'Sell',
            'Hold'  # Hold otherwise
        )
    )

    # Run the simulation
    data, capital = run_simulation(data)

    # Plot the data
    plt.figure(figsize=(14, 7))

    # Subplot 1: Price action with trading signals
    plt.subplot(2, 1, 1)
    plt.plot(data['Date'], data['Close'], label='Close Price', alpha=0.7)
    plt.scatter(data['Date'][data['Action'] == 'Buy'], data['Close'][data['Action'] == 'Buy'],
                label='Buy', color='red', alpha=0.8, s=50)
    plt.scatter(data['Date'][data['Action'] == 'Sell'], data['Close'][data['Action'] == 'Sell'],
                label='Sell', color='green', alpha=0.8, s=50)
    plt.title('Price Action with Trading Signals')
    plt.legend()

    # Subplot 2: Profit over time
    plt.subplot(2, 1, 2)
    plt.plot(data['Date'], data['Profit'], label='Profit', color='blue', alpha=0.7)
    plt.title('Profit Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save the output
    output_file = 'trading_signals_with_profit.csv'
    data.to_csv(output_file, index=False)

    print(f"Trading signals and profit saved to {output_file}")
    print(f"Final capital: {capital:.2f}")

    recommendations = obtain_final_predictions('data.csv')
    print("Final recommendations:", recommendations, len(recommendations))
    print("Actions:", data['Action'].value_counts(), len(data['Action']))

if __name__ == '__main__':
    main('data.csv')
