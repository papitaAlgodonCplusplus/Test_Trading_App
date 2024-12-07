from analyzers.indicators import get_conditions
import traceback
import pandas as pd
from termcolor import colored

def define_actions(data, probabilities, threshold=0.5, analysis_level=0, zigzag=False, context_window=None, patience=20, stop_loss=0.2, take_profit=0.5):
    """Define trading actions based on indicators."""
    take_profit = take_profit / 100
    stop_loss = stop_loss / 100
    data_backup = data.copy()
    if probabilities is not None:
        probabilities_backup = pd.Series(probabilities.copy())

    bullish, bearish, sell_short, buy_to_cover = get_conditions(data_backup, analysis_level)
    data_backup['Action'] = 'Hold'
    data_backup.loc[bullish, 'Action'] = 'Buy'
    data_backup.loc[~bullish & bearish, 'Action'] = 'Sell'
    data_backup.loc[sell_short, 'Action'] = 'Sell Short'
    data_backup.loc[buy_to_cover, 'Action'] = 'Buy to Cover'

    if threshold > 0.0 and probabilities is not None:
        data_backup.loc[(data_backup['Action'] == 'Hold') & (probabilities_backup[:len(data_backup)] > 1 - threshold), 'Action'] = 'Buy'
        data_backup.loc[(data_backup['Action'] == 'Hold') & (probabilities_backup[:len(data_backup)] < threshold), 'Action'] = 'Sell'

    for i in data_backup.index:
        action = data_backup.loc[i, 'Action']
        if action in ['Buy', 'Sell Short']:
            entry_price = data_backup.loc[i, 'Close']
            for j in range(i + 1, min(i + patience + 1, len(data_backup))):
                current_price = data_backup.loc[j, 'Close']
                if action == 'Buy':
                    if current_price >= entry_price * (1 + take_profit):
                        data_backup.loc[j, 'Action'] = 'Sell'
                        break
                    elif current_price <= entry_price * (1 - stop_loss):
                        data_backup.loc[j, 'Action'] = 'Sell'
                        break
                elif action == 'Sell Short':
                    if current_price <= entry_price * (1 - take_profit):
                        data_backup.loc[j, 'Action'] = 'Buy to Cover'
                        break
                    elif current_price >= entry_price * (1 + stop_loss):
                        data_backup.loc[j, 'Action'] = 'Buy to Cover'
                        break
            else: 
                if action == 'Buy':
                    data_backup.loc[j - 1, 'Action'] = 'Sell'
                elif action == 'Sell Short':
                    data_backup.loc[j - 1, 'Action'] = 'Buy to Cover'

    try:
        if 'Action' not in data.columns or data['Action'].empty:
            data['Action'] = 'Hold'
        data.loc[data_backup.index, 'Action'] = data_backup['Action']
    except:
        traceback.print_exc()
        print("Error occurred while defining actions.")

    return data
