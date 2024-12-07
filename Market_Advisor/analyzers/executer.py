from data_helpers.misc import print_colored_sentence
def calculate_take_profit_stop_loss(entry_price, expected_ratio, stop_loss_percentage, inverse=False):
    """
    Calculate the take-profit and stop-loss prices required to match a given risk/reward ratio.

    Parameters:
    - entry_price (float): The price at which the trade is entered.
    - expected_ratio (float): The desired risk/reward ratio.
    - stop_loss_percentage (float): The stop loss percentage (as a positive number).
    - inverse (bool): If True, calculate for inverse logic (fall for take profit, rise for stop loss).

    Returns:
    - float: The take-profit price.
    - float: The stop-loss price.
    """
    if inverse:
        take_profit_price = entry_price * (1 + stop_loss_percentage / 100)
        risk = abs(take_profit_price - entry_price)
        stop_loss_price = entry_price - risk * expected_ratio
    else:
        stop_loss_price = entry_price * (1 - stop_loss_percentage / 100)
        risk = abs(entry_price - stop_loss_price)
        take_profit_price = entry_price + risk * expected_ratio

    return take_profit_price, stop_loss_price

def read_recommendations():
    with open("recommended_actions.txt", 'r') as f:
        last_recommended_action = f.readlines()[-1:]
        last_recommended_action = last_recommended_action[0] if last_recommended_action else None
    return last_recommended_action

def set_sell_action(data, last_action, last_recommended_action, rise, capital, _):
    data_point = data.loc[int(len(data) - 1)]
    data.loc[int(len(data) - 1), 'Action'] = 'Sell' if last_action is not None and last_action != 'Sell' and last_recommended_action is None else 'Hold'
    data.loc[int(len(data) - 1), 'Date'] = data_point['Date']
    data.loc[int(len(data) - 1), 'Open'] = data_point['Open']
    data.loc[int(len(data) - 1), 'High'] = data_point['High']
    data.loc[int(len(data) - 1), 'Low'] = data_point['Low']
    data.loc[int(len(data) - 1), 'Close'] = data_point['Close']
    data.loc[int(len(data) - 1), 'Volume'] = data_point['Volume']
    data_point = data.loc[int(len(data) - 1)]
    with open('recommended_actions.txt', 'w') as file:
        pass
    capital += rise
    return capital

def init_simulation(initial_capital=10000, risk_amount=0.1, risk_percentage=0.01):
    capital = initial_capital
    rr_ratios = []
    profits = []
    last_action = None
    last_recommended_action = read_recommendations()
    entry_price = None 
    entry_price_short = None
    price = None
    risk = risk_percentage
    take_profit_price = None
    stop_loss_price = None
    take_profit_price_short = None
    stop_loss_price_short = None
    risk_amount = (initial_capital * risk_amount) / 100
    return capital, rr_ratios, profits, last_action, last_recommended_action, entry_price, entry_price_short, price, risk, take_profit_price, stop_loss_price, take_profit_price_short, stop_loss_price_short, risk_amount

def calculate_loss_inverse(entry_price, price, stop_loss_price):
    """
    Calculate a value between 0 and 1 based on price, entry_price, and stop_loss_price.
    
    Returns 0 when price equals entry_price, and 1 when price equals stop_loss_price,
    assuming stop_loss_price is greater than entry_price.

    Parameters:
        entry_price (float): The price at which the entry occurred.
        price (float): The current price.
        stop_loss_price (float): The stop loss price.

    Returns:
        float: A value from 0 to 1.
    """
    if price <= entry_price:
        return 0 
    elif price >= stop_loss_price:
        return 1
    else:
        return (price - entry_price) / (stop_loss_price - entry_price)

def calculate_loss(entry_price, price, stop_loss_price):
    """
    Calculate a value between -1 and 0 based on price, entry_price, and stop_loss_price.

    Parameters:
        entry_price (float): The price at which the entry occurred.
        price (float): The current price.
        stop_loss_price (float): The stop loss price.

    Returns:
        float: A value from -1 to 0, where 0 is when price equals entry_price and -1 when price equals stop_loss_price.
    """
    # Ensure the value is within bounds [-1, 0]
    if price >= entry_price:
        return 0  # Price is at or above entry, return 0
    elif price <= stop_loss_price:
        return -1  # Price is at or below stop loss, return -1
    else:
        return (price - entry_price) / (stop_loss_price - entry_price)

def calculate_positive_reward(target_rr_ratio, entry_price, price, take_profit_price):
    """
    Normalize the risk-reward ratio to a value between 0 and target_rr_ratio,
    where 0 corresponds to price = entry_price and target_rr_ratio corresponds to price = take_profit_price.

    Parameters:
        target_rr_ratio (float): The target risk-reward ratio.
        entry_price (float): The entry price.
        price (float): The current price.
        take_profit_price (float): The maximum take-profit price.

    Returns:
        float: The normalized risk-reward value.
    """
    if take_profit_price == entry_price:  # Avoid division by zero
        raise ValueError("Take profit price and entry price must be different.")
    normalized_value = ((price - entry_price) / (take_profit_price - entry_price)) * target_rr_ratio
    return min(max(normalized_value, 0), target_rr_ratio)

def reverse_action(action):
    action = 'Buy' if action == 'Sell' else 'Sell' if action == 'Buy' else action
    action = 'Sell Short' if action == 'Buy to Cover' else 'Buy to Cover' if action == 'Sell Short' else action
    return action

def handle_long_win(data, entry_price, price, take_profit_price, stop_loss_price, risk_amount,
i, row, log=False, target_rr_ratio=3, profits=None, rr_ratios=None):
    reward = target_rr_ratio
    rr_ratios.append(reward)
    profits.append(risk_amount * reward if reward > -1 else -risk_amount)
    if log:
        print_colored_sentence(f"Dt: {row['Date']}, EntryPrice: {entry_price:.5f}, ExitPrice: {price:.5f}, Profit: {profits[-1]:.5f}, Reward: {reward:.5f}, StopLoss: {stop_loss_price:.5f}, TakeProfit: {take_profit_price:.5f}")
    return profits[-1]

def handle_short_win(data, entry_price_short, price, take_profit_price_short, stop_loss_price_short, risk_amount, i, row, log=False, target_rr_ratio=3, profits=None, rr_ratios=None):
    reward = target_rr_ratio
    rr_ratios.append(reward)
    profits.append(risk_amount * reward if reward > -1 else -risk_amount)
    if log:
        print_colored_sentence(f"Dt: {row['Date']}, EntryPrice: {entry_price_short:.5f}, ExitPrice: {price:.5f}, Profit: {profits[-1]:.5f}, Reward: {reward:.5f}, StopLoss: {stop_loss_price_short:.5f}, TakeProfit: {take_profit_price_short:.5f}")
    return profits[-1]

def handle_short_loss(data, entry_price_short, price, take_profit_price_short, stop_loss_price_short, risk_amount, i, row, log=False, target_rr_ratio=3, profits=None, rr_ratios=None):
    reward = -1
    rr_ratios.append(reward)
    profits.append(-risk_amount)
    if log:
        print_colored_sentence(f"Dt: {row['Date']}, EntryPrice: {entry_price_short:.5f}, ExitPrice: {price:.5f}, Profit: {profits[-1]:.5f}, Reward: {reward:.5f}, StopLoss: {stop_loss_price_short:.5f}, TakeProfit: {take_profit_price_short:.5f}")
    return profits[-1]

def handle_long_loss(data, entry_price, price, take_profit_price, stop_loss_price, risk_amount,
i, row, log=False, target_rr_ratio=3, profits=None, rr_ratios=None):
    reward = -1
    rr_ratios.append(reward)
    profits.append(-risk_amount)
    if log:
        print_colored_sentence(f"Dt: {row['Date']}, EntryPrice: {entry_price:.5f}, ExitPrice: {price:.5f}, Profit: {profits[-1]:.5f}, Reward: {reward:.5f}, StopLoss: {stop_loss_price:.5f}, TakeProfit: {take_profit_price:.5f}")
    return profits[-1]

def handle_buy(data, entry_price, price, risk_amount, i, capital, profits, rr_ratios, target_rr_ratio, risk):
    take_profit_price, stop_loss_price = calculate_take_profit_stop_loss(entry_price, target_rr_ratio, risk, inverse=False)
    profits.append(0) 
    rr_ratios.append(0) 
    return take_profit_price, stop_loss_price
    
def handle_sell(data, entry_price, price, risk_amount, i, row, capital, profits, rr_ratios, target_rr_ratio, take_profit_price, stop_loss_price, log=False):
    if price >= take_profit_price:
        reward = target_rr_ratio
    elif price >= entry_price:
        reward = calculate_positive_reward(target_rr_ratio, entry_price=entry_price, price=price, take_profit_price=take_profit_price)
    else:
        reward = -calculate_loss(entry_price=entry_price, price=price, stop_loss_price=stop_loss_price) if price >= stop_loss_price else -1
    rr_ratios.append(reward)
    profits.append(risk_amount * reward if reward > -1 else -risk_amount)
    if log:
        print_colored_sentence(f"Dt: {row['Date']}, EntryPrice: {entry_price:.5f}, ExitPrice: {price:.5f}, Profit: {profits[-1]:.5f}, Reward: {reward:.5f}, StopLoss: {stop_loss_price:.5f}, TakeProfit: {take_profit_price:.5f}")
    return profits[-1]
    
def handle_sell_short(data, entry_price_short, price, risk_amount, i, capital, profits, rr_ratios, target_rr_ratio, risk, stop_loss_price, take_profit_price, log=False):
    stop_loss_price, take_profit_price = calculate_take_profit_stop_loss(entry_price_short, target_rr_ratio, risk, inverse=True)
    profits.append(0) 
    rr_ratios.append(0) 
    return take_profit_price, stop_loss_price

def handle_buy_to_cover(data, entry_price_short, price, risk_amount, i, row, capital, profits, rr_ratios, target_rr_ratio, risk, stop_loss_price, take_profit_price, log=False):
    if price <= take_profit_price:
        reward = target_rr_ratio
    if price <= entry_price_short:
        reward =  calculate_positive_reward(target_rr_ratio, entry_price=entry_price_short, price=price, take_profit_price=take_profit_price)
    else:
        reward = -calculate_loss_inverse(entry_price=entry_price_short, price=price, stop_loss_price=stop_loss_price) if price <= stop_loss_price else -1
    rr_ratios.append(reward)
    profits.append(risk_amount * reward if reward > -1 else -risk_amount)
    if log:
        print_colored_sentence(f"Dt: {row['Date']}, EntryPrice: {entry_price_short:.5f}, ExitPrice: {price:.5f}, Profit: {profits[-1]:.5f}, Reward: {reward:.5f}, StopLoss: {stop_loss_price:.5f}, TakeProfit: {take_profit_price:.5f}")
    return profits[-1]

def run_simulation(data, initial_capital=10000, reverse=False, risk_amount=0.1, target_rr_ratio=3, expected_profit=None, log=False, risk_percentage=0.01):
    """
    Run a trading simulation with Risk/Reward ratio evaluation.
    """
    capital, rr_ratios, profits, last_action, last_recommended_action, entry_price, entry_price_short, price, risk, take_profit_price, stop_loss_price, take_profit_price_short, stop_loss_price_short, risk_amount = init_simulation(initial_capital, risk_amount, risk_percentage)
    
    data['Take Profit Long'] = 0
    data['Stop Loss Long'] = 0
    data['Take Profit Short'] = 0
    data['Stop Loss Short'] = 0
    data["Take Profit Short"] = data["Take Profit Short"].astype(float)
    data["Stop Loss Short"] = data["Stop Loss Short"].astype(float)
    data["Take Profit Long"] = data["Take Profit Long"].astype(float)
    data["Stop Loss Long"] = data["Stop Loss Long"].astype(float)
    data['Vault'] = data['Vault'].astype(float)
    for i, row in data.iterrows():
        action, price = row['Action'], row['Close']
        if reverse:
            action = reverse_action(action)
        if action == "Sell" and not entry_price:
            data.loc[i, 'Action'] = 'Sell_Short'
        if action == "Buy to Cover" and not entry_price_short:
            data.loc[i, 'Action'] = 'Buy'
        if log:
            print_colored_sentence(f"Dt: {row['Date']}, Action: {action}, Price: {price:.5f}")
        if entry_price is not None and price >= take_profit_price:
            data.loc[i, 'Action'] = 'Sell'
            capital_diff = handle_long_win(data, entry_price, price, take_profit_price, stop_loss_price, risk_amount, i, row, log, target_rr_ratio, profits, rr_ratios)
            entry_price = None
            capital += capital_diff
        if entry_price_short is not None and price <= take_profit_price_short:
            data.loc[i, 'Action'] = 'Buy to Cover'
            capital_diff = handle_short_win(data, entry_price_short, price, take_profit_price_short, stop_loss_price_short, risk_amount, i, row, log, target_rr_ratio, profits, rr_ratios)
            entry_price_short = None
            capital += capital_diff
        elif entry_price_short is not None and price >= stop_loss_price_short:
            data.loc[i, 'Action'] = 'Buy to Cover'
            capital_diff = handle_short_loss(data, entry_price_short, price, take_profit_price_short, stop_loss_price_short, risk_amount, i, row, log, target_rr_ratio, profits, rr_ratios)
            entry_price_short = None
            capital += capital_diff
        elif entry_price is not None and price <= stop_loss_price:
            data.loc[i, 'Action'] = 'Sell'
            capital_diff = handle_long_loss(data, entry_price, price, take_profit_price, stop_loss_price, risk_amount, i, row, log, target_rr_ratio, profits, rr_ratios)
            entry_price = None
            capital += capital_diff
        elif action == 'Sell' and entry_price:
            data.loc[i, 'Action'] = 'Sell'
            capital_diff = handle_sell(data, entry_price, price, risk_amount, i, row, capital, profits, rr_ratios, target_rr_ratio, take_profit_price, stop_loss_price, log)
            entry_price = None
            capital += capital_diff
        elif action == 'Buy' and entry_price is None:
            data.loc[i, 'Action'] = 'Buy'
            entry_price = price
            take_profit_price, stop_loss_price = handle_buy(data, entry_price, price, risk_amount, i, capital, profits, rr_ratios, target_rr_ratio, risk)
            capital -= risk_amount
            data['Take Profit Long'] = data['Take Profit Long'].astype(float)
            data['Stop Loss Long'] = data['Stop Loss Long'].astype(float)
            data.loc[i, 'Take Profit Long'] = float(take_profit_price)
            data.loc[i, 'Stop Loss Long'] = float(stop_loss_price)
        elif action == 'Sell Short' and entry_price_short is None:
            data.loc[i, 'Action'] = 'Sell Short'
            entry_price_short = price
            take_profit_price_short, stop_loss_price_short = handle_sell_short(data, entry_price_short, price, risk_amount, i, capital, profits, rr_ratios, target_rr_ratio, risk, stop_loss_price_short, take_profit_price_short, log)
            capital -= risk_amount
            data.loc[i, 'Take Profit Short'] = take_profit_price_short
            data.loc[i, 'Stop Loss Short'] = stop_loss_price_short
        elif action == 'Buy to Cover' and entry_price_short:
            data.loc[i, 'Action'] = 'Buy to Cover'
            capital_diff = handle_buy_to_cover(data, entry_price_short, price, risk_amount, i, row, capital, profits, rr_ratios, target_rr_ratio, risk, stop_loss_price_short, take_profit_price_short, log)
            entry_price_short = None
            capital += capital_diff
        else:
            profits.append(0) 
            rr_ratios.append(0)  
            data.loc[i, 'Action'] = 'Hold'
      
        if expected_profit is not None:
            profit_summatory = sum(profits)
            if profit_summatory >= expected_profit:
                return data, capital

    data.loc[:, 'Risk/Reward Ratios'] = rr_ratios[:len(data)]  
    data.loc[:, 'Profit'] = profits[:len(data)]
    data['Profit'] = data['Profit'].replace(float('inf'), 0)
    return data, capital