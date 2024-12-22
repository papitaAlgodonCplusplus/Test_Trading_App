import pandas as pd
from analyzers.market_predicter import obtain_final_predictions
from analyzers.forex_technical_indicators import forex_indicate
from datetime import datetime
import numpy as np
from termcolor import colored
import MetaTrader5 as mt5

def handle_trailing_stop_loss(data, code_multiplier):
    last_long_entry_price = data.loc[data['Action'] == 'Buy', 'Close'].iloc[-1] if not data.loc[data['Action'] == 'Buy', 'Close'].empty else None
    last_short_entry_price = data.loc[data['Action'] == 'Sell Short', 'Close'].iloc[-1] if not data.loc[data['Action'] == 'Sell Short', 'Close'].empty else None
    current_price = data['Close'].iloc[-1]
    last_buy_action_index = data.loc[data['Action'] == 'Buy'].index[-1] if not data.loc[data['Action'] == 'Buy'].empty else None
    last_sell_action_index = data.loc[data['Action'] == 'Sell Short'].index[-1] if not data.loc[data['Action'] == 'Sell Short'].empty else None
    if last_buy_action_index is not None and last_sell_action_index is not None and last_buy_action_index > last_sell_action_index:
        price_diff = current_price - last_long_entry_price
        current_stop_loss = data.loc[len(data) - 1, 'Stop Loss Long']
        if price_diff > 0 and current_stop_loss < current_price - price_diff:
            change_stop_loss("111111", last_long_entry_price + price_diff, code_multiplier)
            data.loc[len(data) - 1, 'Stop Loss Long'] = last_long_entry_price + price_diff
    last_sell_short_action_index = data.loc[data['Action'] == 'Sell Short'].index[-1] if not data.loc[data['Action'] == 'Sell Short'].empty else None
    last_buy_to_cover_action_index = data.loc[data['Action'] == 'Buy to Cover'].index[-1] if not data.loc[data['Action'] == 'Buy to Cover'].empty else None
    if last_sell_short_action_index is not None and last_buy_to_cover_action_index is not None and last_sell_short_action_index > last_buy_to_cover_action_index:
        price_diff = current_price - last_short_entry_price
        current_stop_loss = data.loc[len(data) - 1, 'Stop Loss Short']
        if price_diff < 0 and current_stop_loss > current_price - price_diff:
            change_stop_loss("111222", last_short_entry_price + price_diff, code_multiplier)
            data.loc[len(data) - 1, 'Stop Loss Short'] = last_short_entry_price + price_diff

def check_position_by_magic(magic_number, log_file):
    # Get all open positions
    positions = mt5.positions_get()
    
    if not positions:
        print_red_blue("No open positions, holding.")
        open(log_file, 'w').close()
        return
    
    # Check if there are positions with the specified magic number
    position_found = False
    for position in positions:
        if position.magic == magic_number:
            print_colored_sentence(f"Position encountered with magic number: {position.magic}")
            position_found = True
    
    # If no position matches the magic number, create an empty file
    if not position_found:
        print_red_blue(f"No position with magic number {magic_number} found. Creating empty log file.")
        open(log_file, 'w').close()
        
def get_current_timestamp():
    """Returns the current date and time (excluding seconds) as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def log_action(action, LOG_FILE):
    """Logs the action with the current date/hour to the log file."""
    timestamp = get_current_timestamp()
    with open(LOG_FILE, "a") as file:
        file.write(f"{timestamp} - {action}\n")

def has_recent_action(current_action, LOG_FILE):
    """Checks if the current date/hour exists in the log file and that the last action is not equal to the current action."""
    timestamp = get_current_timestamp()
    
    try:
        with open(LOG_FILE, "r") as file:
            lines = file.readlines()
            
            # Check if the current timestamp exists in the file
            for line in lines:
                if timestamp in line:
                    print_red_blue(f"An action has already been sent at {timestamp}. Skipping action.")
                    return True
            
            # Check if the last logged action is the same as the current action
            if lines:
                last_line = lines[-1].strip()
                if last_line.endswith(current_action):
                    print_red_blue(f"Last action was the same as the current action: {current_action}. Skipping action.")
                    return True
                
    except FileNotFoundError:
        return False

    return False

def send_action_to_mt5(data, symbol, stop_loss_pips=5, take_profit_pips=5, trailing_stop_pips=10, code_multiplier=1, log_file=None):
    last_action = data["Action"].iloc[-1]

    LOG_FILE = "temp_files/last_action_log.txt"
    if log_file is not None:
        LOG_FILE = log_file
    if has_recent_action(last_action, LOG_FILE):
        return

    lots = 0.5

    if last_action == "Buy":
        price = mt5.symbol_info_tick(symbol).ask
        stop_loss = price - (0.0001) * stop_loss_pips
        take_profit = price + (0.0001) * take_profit_pips
        trailing_stop = int(trailing_stop_pips * 10)  # MT5 uses points (1 pip = 10 points)
        
        buy_result = place_buy_order(symbol, lots, price, stop_loss, take_profit, code_multiplier)
        print_colored_sentence(f"Buy Order Result: {buy_result}")
        log_action("Buy", LOG_FILE)
        mt5.shutdown()

    elif last_action == "Sell Short":
        price = mt5.symbol_info_tick(symbol).bid
        stop_loss = price + (0.0001) * stop_loss_pips
        take_profit = price - (0.0001) * take_profit_pips
        trailing_stop = int(trailing_stop_pips * 10)  # MT5 uses points (1 pip = 10 points)

        sell_result = place_sell_order(symbol, lots, price, stop_loss, take_profit, code_multiplier)
        print_colored_sentence(f"Sell Short Order Result: {sell_result}")
        log_action("Sell Short", LOG_FILE)
        mt5.shutdown()

    elif last_action == "Sell":
        tick_info = mt5.symbol_info_tick(symbol)
        if tick_info is None:
            print(f"Failed to get tick info for symbol: {symbol}, {mt5.last_error()}")
            return
        volume = lots
        close_position("111111", volume, code_multiplier)
        log_action("Sell", LOG_FILE)
        mt5.shutdown()

    elif last_action == "Buy to Cover":
        volume = lots
        close_position("111222", volume, code_multiplier)
        log_action("Buy to Cover", LOG_FILE)
        mt5.shutdown()

def calculate_partial_volume(entry_price, current_price, position_volume, close_percentage_per_pip=0.05):
    """
    Calculate the partial volume to close based on the difference between entry and current price.
    Parameters:
    - entry_price (float): The entry price of the position.
    - current_price (float): The current price of the symbol.
    - position_volume (float): The total volume of the position.
    - close_percentage_per_pip (float): The percentage of the position to close per pip difference (default: 0.05).
    Returns:
    - float: The calculated partial volume to close.
    """
    price_diff = abs(current_price - entry_price)
    price_diff_pips = price_diff / 0.00010
    close_percentage = price_diff_pips * close_percentage_per_pip
    close_volume = position_volume * close_percentage
    close_volume = round(max(close_volume, 0.01), 2)
    close_volume = min(close_volume, position_volume)
    return close_volume

import MetaTrader5 as mt5

def change_stop_loss(magic_number, new_stop_loss, code_multiplier=1):
    positions = mt5.positions_get()
    if not positions:
        print_colored_sentence("No open positions found.")
        return False

    # Adjust magic number based on the multiplier
    magic_number = int(magic_number)
    if code_multiplier != 1:
        magic_number *= code_multiplier

    for position in positions:
        print_colored_sentence(f"MAGIC: {position.magic}, {magic_number}")
        if position.magic == magic_number:
            print_colored_sentence(f"Found position to update stop loss: Ticket {position.ticket}, Current Stop Loss {position.sl}")

            # Create stop loss change request
            sl_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "symbol": position.symbol,
                "sl": new_stop_loss,
                "tp": position.tp,  # Keep the current take profit unchanged
                "magic": magic_number,
                "comment": f"Change stop loss to {new_stop_loss}",
            }

            print_colored_sentence(f"Sending stop loss change request: {sl_request}")

            # Send the order
            result = mt5.order_send(sl_request)
            print_colored_sentence(f"OrderSend Result: {result}")

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print_colored_sentence(f"Successfully updated stop loss for position {position.ticket} to {new_stop_loss}")
                return True
            else:
                print_colored_sentence(f"Failed to update stop loss for position {position.ticket}: {result.retcode}, {result.comment}")
                return False

    print_colored_sentence(f"No positions found with magic number {magic_number}")
    return False

def close_position(magic_number, close_volume, code_multiplier=1):
    positions = mt5.positions_get()
    if not positions:
        print_colored_sentence("No open positions found.")
        return False

    # Adjust magic number based on the multiplier
    magic_number = int(magic_number)
    if code_multiplier != 1:
        magic_number *= code_multiplier

    for position in positions:
        print_colored_sentence(f"MAGIC: {position.magic}, {magic_number}")
        if position.magic == magic_number:
            print_colored_sentence(f"Found position to partially close: Ticket {position.ticket}, Volume {position.volume}")

            # Validate close volume
            if close_volume > position.volume:
                print_colored_sentence(f"Close volume ({close_volume}) exceeds position volume ({position.volume}).")
                return False

            # Create close request
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": position.ticket,
                "symbol": position.symbol,
                "volume": close_volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 10,
                "magic": magic_number,
                "comment": f"Partial close of {close_volume} lots",
            }

            print_colored_sentence(f"Sending close request: {close_request}")

            # Send the order
            result = mt5.order_send(close_request)
            print_colored_sentence(f"OrderSend Result: {result}")

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print_colored_sentence(f"Successfully closed {close_volume} lots of position {position.ticket}")
                # Verify remaining volume
                updated_positions = mt5.positions_get(ticket=position.ticket)
                if updated_positions:
                    remaining_volume = updated_positions[0].volume
                    print_colored_sentence(f"Remaining volume for position {position.ticket}: {remaining_volume}")
                else:
                    print_colored_sentence(f"Position {position.ticket} no longer exists (fully closed).")
                return True
            else:
                print_colored_sentence(f"Failed to close position {position.ticket}: {result.retcode}, {result.comment}")
                return False

    print_colored_sentence(f"No positions found with magic number {magic_number}")
    return False


def place_buy_order(symbol, volume, price, stop_loss, take_profit, code_multiplier):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": 10,
        "magic": 111111 * code_multiplier,
        "comment": "Python Buy Order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)
    if result is None:
        print(f"Order send failed: {mt5.last_error()}")
        return None
    return result

def place_sell_order(symbol, volume, price, stop_loss, take_profit, code_multiplier):
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select symbol: {symbol}")
        return None
    if volume <= 0:
        print("Volume must be greater than 0.")
        return None
    if price <= 0 or stop_loss <= 0 or take_profit <= 0:
        print("Price, stop loss, and take profit must be positive values.")
        return None
    request = {
        "action": mt5.ORDER_TYPE_SELL_LIMIT,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": 10,
        "magic": 111222 * code_multiplier,
        "comment": "Python Sell Order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    print(f"Sending order request: {request}")
    result = mt5.order_send(request)
    if result is None:
        print(f"Order send failed: {mt5.last_error()}")
        return None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed with retcode: {result.retcode}")
        print(f"Error details: {mt5.last_error()}")
    else:
        print(f"Order successful, result: {result}")
    return result

def check_for_patience(data, patience):
    last_action = data.loc[len(data) - 1, 'Action']
    if last_action == "Sell":
        last_buy_action = data.loc[data['Action'] == "Buy"].tail(1)
        last_buy_action_index = last_buy_action.index[0] if not last_buy_action.empty else 0
        
        stop_loss = data.loc[last_buy_action_index, 'Stop Loss Long'] if not last_buy_action.empty else 0
        take_profit = data.loc[last_buy_action_index, 'Take Profit Long'] if not last_buy_action.empty else 0
        
        close_price = data.loc[len(data) - 1, 'Close']
        current_wait_time = len(data) - last_buy_action_index
        
        if current_wait_time >= patience / 1.5:
            take_profit = take_profit - 0.0002
            data.loc[len(data) - 1, 'Take Profit Long'] = take_profit
        elif current_wait_time >= patience / 2.5:
            take_profit = take_profit - 0.0001
            data.loc[len(data) - 1, 'Take Profit Long'] = take_profit
        
        if current_wait_time <= patience and stop_loss > 0 and close_price >= stop_loss and close_price <= take_profit:
            data.loc[len(data) - 1, 'Action'] = "Hold"
            print_colored_sentence(f"Patience still exists: {last_buy_action_index}, at {len(data)}, as {current_wait_time} <= {patience}")
            
    elif last_action == "Buy to Cover":
        last_sell_short_action = data.loc[data['Action'] == "Sell Short"].tail(1)
        last_sell_short_action_index = last_sell_short_action.index[0] if not last_sell_short_action.empty else 0
        
        stop_loss = data.loc[last_sell_short_action_index, 'Stop Loss Short'] if not last_sell_short_action.empty else 0
        take_profit = data.loc[last_sell_short_action_index, 'Take Profit Short'] if not last_sell_short_action.empty else 0
        
        close_price = data.loc[len(data) - 1, 'Close']
        current_wait_time = len(data) - last_sell_short_action_index
        
        if current_wait_time >= patience / 1.5:
            take_profit = take_profit + 0.0002
            data.loc[len(data) - 1, 'Take Profit Short'] = take_profit
        elif current_wait_time >= patience / 2.5:
            take_profit = take_profit + 0.0001
            data.loc[len(data) - 1, 'Take Profit Short'] = take_profit
        
        if current_wait_time <= patience and stop_loss > 0 and close_price <= stop_loss and close_price >= take_profit:
            data.loc[len(data) - 1, 'Action'] = "Hold"
            print_colored_sentence(f"Patience still exists: {last_sell_short_action_index}, at {len(data)}, as {current_wait_time} <= {patience}")
        else:
            print_colored_sentence(f"Patience ran out: {last_sell_short_action_index}, at {len(data)}, as {current_wait_time} > {patience}")
            
def check_for_expired_actions(data, patience):
    last_buy_action = data.loc[data['Action'] == "Buy"].tail(1)
    last_sell_action = data.loc[data['Action'] == "Sell"].tail(1)
    last_sell_action_index = last_sell_action.index[0] if not last_sell_action.empty else 0
    last_buy_action_index = last_buy_action.index[0] if not last_buy_action.empty else 0
    if last_buy_action_index > last_sell_action_index and len(data) - last_buy_action_index > patience:
        data.loc[len(data) - 1, 'Action'] = "Sell"
        
        return
    
    if 'Stop Loss Long' in data.columns:
        stop_loss = data.loc[last_buy_action_index, 'Stop Loss Long'] if not last_buy_action.empty else 0
        if stop_loss > 0 and data.loc[len(data) - 1, 'Close'] <= stop_loss:
            data.loc[len(data) - 1, 'Action'] = "Sell"
            
            return
        
    last_sell_short_action = data.loc[data['Action'] == "Sell Short"].tail(1)
    last_buy_to_cover_action = data.loc[data['Action'] == "Buy to Cover"].tail(1)
    last_buy_to_cover_action_index = last_buy_to_cover_action.index[0] if not last_buy_to_cover_action.empty else 0
    last_sell_short_action_index = last_sell_short_action.index[0] if not last_sell_short_action.empty else 0
    if last_sell_short_action_index > last_buy_to_cover_action_index and len(data) - last_sell_short_action_index > patience:
        data.loc[len(data) - 1, 'Action'] = "Buy to Cover"
        
        return
    
    if 'Stop Loss Short' in data.columns:
        stop_loss = data.loc[last_sell_short_action_index, 'Stop Loss Short'] if not last_sell_short_action.empty else 0
        if stop_loss > 0 and data.loc[len(data) - 1, 'Close'] >= stop_loss:
            data.loc[len(data) - 1, 'Action'] = "Buy to Cover"
            
            return
    
def check_not_repeated_last_action(data):
    """
    Check the last action in the data. If it's a 'Buy' or 'Sell Short' action,
    ensure the previous corresponding 'Sell' or 'Buy to Cover' action exists
    in the data. If not, set the last action as 'Hold'.
    
    Parameters:
        data (pd.DataFrame): DataFrame with at least a column named 'Action'.
    """
    if len(data) < 2:
        return data
    
    last_action_index = len(data) - 1
    last_action = data.iloc[last_action_index]['Action']
    if last_action in ['Buy', 'Sell Short']:
        for i in range(last_action_index - 1, -1, -1):
            previous_action = data.iloc[i]['Action']
            if (last_action == 'Buy' and previous_action == 'Sell') or \
               (last_action == 'Sell Short' and previous_action == 'Buy to Cover'):
                return
            if previous_action == last_action:
                data.at[last_action_index, 'Action'] = 'Hold'
                return
    elif last_action in ['Sell', 'Buy to Cover']:
        for i in range(last_action_index - 1, -1, -1):
            previous_action = data.iloc[i]['Action']
            if (last_action == 'Sell' and previous_action == 'Buy') or \
               (last_action == 'Buy to Cover' and previous_action == 'Sell Short'):
                return
            if previous_action == last_action:
                data.at[last_action_index, 'Action'] = 'Hold'
                return
    return

def cement(data, cemented_data):
    if cemented_data is not None:
        data.loc[cemented_data.index, "Action"] = cemented_data["Action"]
        data.loc[cemented_data.index, "Profit"] = cemented_data["Profit"]
        data.loc[cemented_data.index, "Risk/Reward Ratios"] = cemented_data["Risk/Reward Ratios"]
        data.loc[cemented_data.index, "Take Profit Long"] = cemented_data["Take Profit Long"]
        data.loc[cemented_data.index, "Stop Loss Long"] = cemented_data["Stop Loss Long"]
        data.loc[cemented_data.index, "Take Profit Short"] = cemented_data["Take Profit Short"].astype(float)
        data.loc[cemented_data.index, "Stop Loss Short"] = cemented_data["Stop Loss Short"].astype(float)
  
def pad_zeros_with_last_nonzero(lst, close_prices):
    padded_list = lst[:]
    padded_list = [float(x) for x in padded_list]  
    last_nonzero = None
    for value in padded_list:
        if value != 0:
            last_nonzero = value
            break
    if last_nonzero is None:
        last_nonzero = float(close_prices[0])
    for i in range(len(padded_list)):
        if padded_list[i] == 0 and last_nonzero is not None:
            padded_list[i] = last_nonzero
        elif padded_list[i] != 0:
            last_nonzero = padded_list[i]
    return padded_list

def pad_take_profits_and_stop_losses(data):
    data['Take Profit Long'] = data['Take Profit Long'].astype(float)
    data['Take Profit Short'] = data['Take Profit Short'].astype(float)
    data['Stop Loss Long'] = data['Stop Loss Long'].astype(float)
    data['Stop Loss Short'] = data['Stop Loss Short'].astype(float)
    
    data.loc[:, 'Take Profit Long'] = pad_zeros_with_last_nonzero(data.loc[:, 'Take Profit Long'], data.loc[:, 'Close'])
    data.loc[:, 'Take Profit Short'] = pad_zeros_with_last_nonzero(data.loc[:, 'Take Profit Short'], data.loc[:, 'Close'])
    data.loc[:, 'Stop Loss Long'] = pad_zeros_with_last_nonzero(data.loc[:, 'Stop Loss Long'], data.loc[:, 'Close'])
    data.loc[:, 'Stop Loss Short'] = pad_zeros_with_last_nonzero(data.loc[:, 'Stop Loss Short'], data.loc[:, 'Close'])

def print_red_blue(sentence):
    words = sentence.split()
    for i, word in enumerate(words):
        color = 'red' if i % 2 == 0 else 'blue'
        print(colored(word, color), end=" ")
    print()
    
def print_colored_sentence(sentence):
    colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta']
    words = sentence.split()
    for i, word in enumerate(words):
        color = colors[i % len(colors)]
        print(colored(word, color), end=" ")
    print()

def calculate_profit_summatory(data):
    data['Profit Summatory'] = data['Profit'].cumsum()

def wins_and_losses_count(data):
    wins, losses, break_evens = 0, 0, 0
    if 'Win/Loss Ratio' not in data.columns:
        data['Win/Loss Ratio'] = 0
    data['Win/Loss Ratio'] = data['Win/Loss Ratio'].astype(float)
    for row in data.itertuples():
        if row.Profit > 0:
            wins += 1
        elif row.Profit < 0:
            losses += 1
        else:
            break_evens += 1
        total = wins + losses
        data.loc[row.Index, 'Win/Loss Ratio'] = (wins - losses) / total if total > 0 else 0

def pad_predictions_and_probabilities(predictions, probabilities, target_length):
    """Ensure predictions and probabilities are of the same length as target."""
    if len(predictions) < target_length:
        predictions = np.pad(predictions, (0, target_length - len(predictions)), constant_values=3)
    if len(probabilities) < target_length:
        probabilities = np.pad(probabilities, (0, target_length - len(probabilities)), constant_values=3)
    return predictions, probabilities

def update_simulation_data(data, data_for_simulation):
    """Update main data with the simulation results."""
    data.loc[data_for_simulation.index, 'Action'] = data_for_simulation['Action']
    data.loc[data_for_simulation.index, 'Profit'] = data_for_simulation['Profit']
    return max(data_for_simulation['Profit'].replace('', np.nan).fillna(0))

def print_results(results):
    current_iteration = 0
    for result in results:
        print(f"\033[91mAnalysis Level: {result['level']} with Risk Amount: {result['risk_amount']:.2f}, and Threshold: {result['threshold']}\033[0m")
        print(f"\033[94mFinal capital: {result['final_capital']:.2f}\033[0m")  
        print(f"\033[92mDebt Profit: {result['debt_profit']:.2f}\033[0m")  
        print("-" * 50)
        current_iteration += 1
        
        if result['debt_profit'] <= 0 or current_iteration >= 3:
            break

def print_actions(data):
    for row in data.itertuples():
        if row.Action == 'Buy':
            print(f"\033[91m{row.Date}: {row.Action} at {row.Close:.2f}\033[0m")
        elif row.Action == 'Sell':
            print(f"\033[92m{row.Date}: {row.Action} at {row.Close:.2f}\033[0m")
        elif row.Action == 'Sell Short':
            print(f"\033[93m{row.Date}: {row.Action} at {row.Close:.2f}\033[0m")
        elif row.Action == 'Buy to Cover':
            print(f"\033[94m{row.Date}: {row.Action} at {row.Close:.2f}\033[0m")
          
def write_to_file(output_file, best_results):
    with open(output_file, 'a') as f:
        f.write(f"\n--- Best Hyperparameters Round ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
        for i, result in enumerate(best_results, start=1):
            f.write(f"Rank {i}:\n")
            f.write(f"  Analysis Level: {result['level']}\n")
            f.write(f"  Threshold: {result['threshold']}\n")
            f.write(f"  Risk Amount: {result['risk_amount']:.2f}\n")
            f.write(f"  Final Capital: {result['final_capital']:.2f}\n")
            f.write(f"  Debt Profit: {result['debt_profit']:.2f}\n")
            f.write(f"  Patience: {result['patience']}\n")
            f.write(f"  Take Profit: {result['take_profit']}\n")
            f.write(f"  Stop Loss: {result['stop_loss']}\n")
            f.write("-" * 50 + "\n")

    print(f"Best hyperparameters saved to {output_file}")

import pandas as pd
import numpy as np

def load_data(
    file_path, 
    reverse=False, 
    multiply_factor=None, 
    initial_capital=50000, 
    deep_calculation=False, 
    bot_type=1,
    web_scraping="5m",
    advisor_threshold=3,
):
    data = pd.read_csv(file_path, parse_dates=['Date'])

    print(f"\033[95m --------- Obtained New Data ---------\033[0m")
    print(f"\033[95mLast date in dataset: {data['Date'].iloc[-1]}\033[0m")
    print(f"\033[95m -------------------------------------\033[0m")

    if multiply_factor is not None:
        data['Close'] = data['Close'] * multiply_factor
        data['High'] = data['High'] * multiply_factor
        data['Low'] = data['Low'] * multiply_factor
        data['Open'] = data['Open'] * multiply_factor

    predictions, probabilities = None, None

    bot_configs = {
        1: {
            'analysis_levels': [0],
            'threshold_options': [0.0],
            'risk_options': [5],
            'patience_options': [50],
            'take_profit_options': [0.1],
            'stop_loss_options': [0.01],
        },
        2: {
            'analysis_levels': [4],
            'threshold_options': [0.0],
            'risk_options': [5],
            'patience_options': [50],
            'take_profit_options': [0.05],
            'stop_loss_options': [0.01],
        },
        3: {
            'analysis_levels': [3],
            'threshold_options': [0.0],
            'risk_options': [5],
            'patience_options': [100],
            'take_profit_options': [0.2],
            'stop_loss_options': [0.025],
        },
        4: {
            'analysis_levels': [1],
            'threshold_options': [0.0],
            'risk_options': [5],
            'patience_options': [100],
            'take_profit_options': [0.1],
            'stop_loss_options': [0.025],
        },
        5: {
            'analysis_levels': [2],
            'threshold_options': [0.0],
            'risk_options': [5],
            'patience_options': [150],
            'take_profit_options': [0.1],
            'stop_loss_options': [0.05],
        },
        6: {
            'analysis_levels': [5],
            'threshold_options': [0.0],
            'risk_options': [5],
            'patience_options': [100],
            'take_profit_options': [0.1],
            'stop_loss_options': [0.01],
        },
        7: {
            'analysis_levels': [6],
            'threshold_options': [0.0],
            'risk_options': [5],
            'patience_options': [100],
            'take_profit_options': [0.1],
            'stop_loss_options': [0.01],
        },
        8: {
            'analysis_levels': [7],
            'threshold_options': [0.0],
            'risk_options': [5],
            'patience_options': [100],
            'take_profit_options': [0.1],
            'stop_loss_options': [0.01],
        },
    }

    summatory_of_indicators = forex_indicate(file_path, web_scraping)
    
    if summatory_of_indicators >= advisor_threshold:
        bot_type = 8
    elif summatory_of_indicators <= -advisor_threshold:
        bot_type = 7
    
    if not deep_calculation:
        config = bot_configs.get(bot_type, bot_configs[1])
        analysis_levels = config['analysis_levels']
        threshold_options = config['threshold_options']
        risk_options = config['risk_options']
        patience_options = config['patience_options']
        take_profit_options = config['take_profit_options']
        stop_loss_options = config['stop_loss_options']
    else:
        analysis_levels = [0, 1, 2, 3, 4]
        threshold_options = np.arange(0.0, 0.4, 0.1)
        risk_options = [1, 5]
        patience_options = [50, 150, 300, 500]
        take_profit_options = [0.01, 0.05, 0.1, 0.15, 0.2]
        stop_loss_options = [0.001, 0.005, 0.01, 0.015, 0.02]

    output_file = "best_hyperparameters.txt"

    if 'Action' not in data.columns:
        data['Action'] = 'Hold'

    if len(data) > 50 and threshold_options[0] != 0.0:
        predictions, probabilities = obtain_final_predictions(file_path)
        if reverse:
            predictions = 1 - predictions
            
        if len(predictions) < len(data):
            predictions = np.pad(predictions, (0, len(data) - len(predictions)), constant_values=3)

        if len(probabilities) < len(data):
            probabilities = np.pad(probabilities, (0, len(data) - len(probabilities)), constant_values=3)

    return (
        data, 
        predictions, 
        probabilities, 
        analysis_levels, 
        threshold_options, 
        risk_options, 
        patience_options, 
        take_profit_options, 
        stop_loss_options, 
        initial_capital, 
        output_file,
        summatory_of_indicators
    )
