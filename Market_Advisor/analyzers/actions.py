from analyzers.indicators import get_conditions
import traceback
from termcolor import colored

def print_colored_sentence(sentence):
    colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta']
    words = sentence.split()
    for i, word in enumerate(words):
        color = colors[i % len(colors)]
        print(colored(word, color), end=" ")
    print()

def define_actions(data, probabilities, threshold=0.5, analysis_level=0, zigzag=False, context_window=None, last_written_row=0):
    """Define trading actions based on indicators."""
    
    data_backup = data.copy()
    probabilities_backup = probabilities.copy()
    if context_window is not None:
        try:
            data_backup = data_backup.iloc[:context_window]
            probabilities_backup = probabilities[:context_window]
        except:
            traceback.print_exc()
            print("Error occurred while defining actions.")

import pandas as pd
from termcolor import colored

import pandas as pd
import traceback
def define_actions(data, probabilities, threshold=0.5, analysis_level=0, zigzag=False, context_window=None, patience=20, stop_loss=0.2, take_profit=0.5, reverse=False):
    """Define trading actions based on indicators, enforcing patience for consecutive holds."""
    data_backup = data.copy()
    if probabilities is not None:
        probabilities_backup = pd.Series(probabilities.copy())

    buy_long, sell_long, sell_short, buy_to_cover = get_conditions(data_backup, analysis_level, reverse=reverse)
    data_backup['Action'] = 'Hold'
    data_backup.loc[buy_long, 'Action'] = 'Buy'
    data_backup.loc[sell_long, 'Action'] = 'Sell'
    data_backup.loc[sell_short, 'Action'] = 'Sell Short'
    data_backup.loc[buy_to_cover, 'Action'] = 'Buy to Cover'

    if threshold > 0.0 and probabilities is not None:
        data_backup.loc[(data_backup['Action'] == 'Hold') & (probabilities_backup[:len(data_backup)] > 1 - threshold), 'Action'] = 'Buy'
        data_backup.loc[(data_backup['Action'] == 'Hold') & (probabilities_backup[:len(data_backup)] < threshold), 'Action'] = 'Sell'

    hold_long_counter = 0
    hold_short_counter = 0
    entry_price_long = 0
    entry_price_short = 0
    holding_buy = False
    holding_sell_short = False

    for i in range(len(data_backup)):
        action = data_backup.loc[i, 'Action']
        current_price = data_backup.loc[i, 'Close']
        
        if holding_buy:
            hold_long_counter += 1
            if hold_long_counter >= patience or current_price - entry_price_long >= take_profit or current_price - entry_price_long <= -stop_loss:
                holding_buy = False
                hold_long_counter = 0
                entry_price_long = 0
                data_backup.loc[i, 'Action'] = 'Sell'
            else:
                data_backup.loc[i, 'Action'] = 'Hold'
        
        elif holding_sell_short:
            hold_short_counter += 1
            if hold_short_counter >= patience or entry_price_short - current_price >= take_profit or entry_price_short - current_price <= -stop_loss:
                holding_sell_short = False
                hold_short_counter = 0
                entry_price_short = 0
                data_backup.loc[i, 'Action'] = 'Buy to Cover'
            else:
                data_backup.loc[i, 'Action'] = 'Hold'
        
        elif action == 'Buy':
            holding_buy = True
            hold_long_counter = 0
            entry_price_long = current_price
        
        elif action == 'Sell Short':
            holding_sell_short = True
            hold_short_counter = 0
            entry_price_short = current_price

    try:
        if 'Action' not in data.columns or data['Action'].empty:
            data['Action'] = 'Hold'
        data.loc[data_backup.index, 'Action'] = data_backup['Action']
    except Exception:
        traceback.print_exc()
        print("Error occurred while defining actions.")

    return data
