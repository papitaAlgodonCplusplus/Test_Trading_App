from termcolor import colored
from itertools import product
from analyzers.actions import define_actions
from tqdm import tqdm
from analyzers.market_predicter import obtain_final_predictions
from data_helpers.misc import pad_predictions_and_probabilities, print_results, print_actions, write_to_file
from analyzers.market_adviser import MarketAdviser
from analyzers.executer import run_simulation
import pandas as pd
import os
import numpy as np

import pandas as pd
from analyzers.market_predicter import obtain_final_predictions
from itertools import product
from analyzers.actions import define_actions
from tqdm import tqdm
from analyzers.market_predicter import obtain_final_predictions
from analyzers.actions import define_actions
from data_helpers.misc import write_to_file, print_results, print_actions, pad_predictions_and_probabilities, print_colored_sentence

def process_simulation(data, probabilities, analysis_levels, threshold_options, risk_options, initial_capital, _, expected_profit, context_window, last_written_row=0):
    """Run simulations for all parameter combinations and return sorted results."""
    results = []
    actions_backup = None
    reverse_options = [False, True]
    total_combinations = len(analysis_levels) * len(threshold_options) * len(risk_options) * len(reverse_options)

    if last_written_row != 0:
        actions_backup = data.iloc[:last_written_row]['Action'].copy()
        
    with tqdm(total=total_combinations, desc="Running simulations") as pbar:
        for level, threshold, risk_amount, reverse_option in product(analysis_levels, threshold_options, risk_options, reverse_options):
            define_actions(data, probabilities, threshold, level, True, context_window=context_window, last_written_row=last_written_row)

            if last_written_row != 0:
                last_date = data.iloc[last_written_row]['Date']
                data.loc[:last_date, 'Action'] = actions_backup

            print_colored_sentence(f"Level: {level}, Threshold: {threshold}, Risk: {risk_amount}, Reverse: {reverse_option}")
            data, capital, profit_vault = run_simulation(data, initial_capital, reverse_option, risk_amount, expected_profit=expected_profit)
            debt_profit = profit_vault - abs(capital - initial_capital)
            results.append({
                "level": level,
                "threshold": threshold,
                "risk_amount": risk_amount,
                "final_capital": capital,
                "profit_vault": profit_vault,
                "debt_profit": debt_profit,
                "reverse_option": reverse_option
            })
            pbar.update(1)
    
    return sorted(results, key=lambda x: x["debt_profit"], reverse=True)

def process_simulation_window(data, probabilities, analysis_levels, threshold_options, risk_options,
                              initial_capital, reverse, expected_profit, context_window, output_file, file_path, last_written_row=0):
    """Process a single simulation window."""
    actions_backup = None
    
    if last_written_row != 0:
        actions_backup = data.iloc[:last_written_row]['Action'].copy()
            
    sorted_results = process_simulation(data, probabilities, analysis_levels, threshold_options, risk_options,
                                        initial_capital, reverse, expected_profit, context_window, last_written_row)
    print_results(sorted_results)

    # Process best results
    best_results = sorted_results[:3]
    result = best_results[0]
    predictions, probabilities = obtain_final_predictions(file_path)
    predictions = 1 - predictions
    predictions, probabilities = pad_predictions_and_probabilities(predictions, probabilities, len(data))

    define_actions(data, probabilities, result['threshold'], result['level'], True, context_window=context_window, last_written_row=last_written_row)
    
    if last_written_row != 0:
        last_date = data.iloc[last_written_row]['Date']
        data.loc[:last_date, 'Action'] = actions_backup
    
    print_colored_sentence(f"FINAL Level: {result['level']}, Threshold: {result['threshold']}, Risk: {result['risk_amount']}, Reverse: {result['reverse_option']}")
    data, _, _ = run_simulation(data, initial_capital, result['reverse_option'], result['risk_amount'], expected_profit=None, log=False)
   
    print_actions(data)
    write_to_file(output_file, best_results)
    return data, predictions, probabilities


def run_simulation(data, initial_capital=50000, reverse=False, risk_amount=None, expected_profit=None, log=False):
    """Run a trading simulation."""
    capital, units_hold, profit_vault = initial_capital, 0, 0
    profit_over_time = []
    vault_values = []
    data.loc[:, 'Action'] = data['Action'].mask(data['Action'] == data['Action'].shift(), 'Hold')

    for i, row in data.iterrows():
        action, price = row['Action'], row['Close']
        if reverse:
            action = 'Buy' if action == 'Sell' else 'Sell' if action == 'Buy' else 'Hold'

        if action == 'Buy' and capital > 0:
            invest = min(risk_amount or capital, capital)
            units_hold = invest / price
            capital -= invest
            data.loc[i, 'Action'] = 'Buy'
        elif action == 'Sell' and units_hold > 0:
            capital += units_hold * price
            units_hold = 0
            data.loc[i, 'Action'] = 'Sell'
        else:
            data.loc[i, 'Action'] = 'Hold'

def call_adviser_for_last_action(data):
    if len(data) < 10:
        return None, None
    n_of_holdings_past_10_days = data['Action'].tail(10).value_counts().get('Hold', 0)
    if n_of_holdings_past_10_days > 5:
        data.loc[:, 'Hold_Streak'] = (data['Action'] != 'Hold').cumsum()
        current_date = data['Date'].iloc[-1]   
        start_date = current_date - pd.DateOffset(days=10)
        finish_date = current_date
        adviser = MarketAdviser()
        columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        sample_data_path = 'data/sample_data.csv'
        data_to_save = data[(data['Date'] >= start_date) & (data['Date'] <= finish_date)][columns]
        data_to_save.to_csv(sample_data_path, index=False)
        recommended_action = adviser.predict_last_recommended_action(sample_data_path)
        try:
            os.remove(sample_data_path)
        except FileNotFoundError:
            pass
        return recommended_action, finish_date

def define_actions_extended(data, cemented_data=None):
    last_recommended_action = None
    with open("recommended_actions.txt", 'r') as f:
        last_recommended_action = f.readlines()[-1:]
        last_recommended_action = last_recommended_action[0] if last_recommended_action else None
    if last_recommended_action is not None:
        last_recommended_action = last_recommended_action.strip()
        data_point = data.loc[int(last_recommended_action)]
        data.loc[int(last_recommended_action), 'Action'] = 'Buy'
        data.loc[int(last_recommended_action), 'Date'] = data_point['Date']
        data.loc[int(last_recommended_action), 'Open'] = data_point['Open']
        data.loc[int(last_recommended_action), 'High'] = data_point['High']
        data.loc[int(last_recommended_action), 'Low'] = data_point['Low']
        data.loc[int(last_recommended_action), 'Close'] = data_point['Close']
        data.loc[int(last_recommended_action), 'Volume'] = data_point['Volume']
        data_point = data.loc[int(last_recommended_action)]
        if cemented_data is not None:
            if len(cemented_data) < int(last_recommended_action):
                data_point = cemented_data.loc[int(last_recommended_action)]
                cemented_data.loc[int(last_recommended_action), 'Action'] = 'Buy'
                cemented_data.loc[int(last_recommended_action), 'Date'] = data_point['Date']
                cemented_data.loc[int(last_recommended_action), 'Open'] = data_point['Open']
                cemented_data.loc[int(last_recommended_action), 'High'] = data_point['High']
                cemented_data.loc[int(last_recommended_action), 'Low'] = data_point['Low']
                cemented_data.loc[int(last_recommended_action), 'Close'] = data_point['Close']
                cemented_data.loc[int(last_recommended_action), 'Volume'] = data_point['Volume']
                data_point = cemented_data.loc[int(last_recommended_action)]
    
    if last_recommended_action is None and len(data) > 15:
        result = call_adviser_for_last_action(data)
        if result is not None:
            recommended_action, recommended_date = result
            if recommended_action == 1:
                index_of_recommended_date = data[data['Date'] == recommended_date].index[0]
                data.loc[index_of_recommended_date, 'Action'] = 'Buy'
                with open("recommended_actions.txt", 'a') as f:
                    f.write(f"{index_of_recommended_date}\n")  

def has_consecutive_holds(actions):
    if not actions or actions[-1] == "Hold":
        return False
    consecutive_holds = 0
    for action in reversed(actions[:-1]):
        if action == "Hold":
            consecutive_holds += 1
        else:
            break
    return consecutive_holds > 10

def process_simulation(data, probabilities, analysis_levels, threshold_options, risk_options, 
                       patience_options, take_profit_options, stop_loss_options,
                       initial_capital, reverse, expected_profit, context_window, cemented_data=None, use_analyzer=True, stop_loss_pips=5, take_profit_pips=5, pip_value=0.0001, target_rr_ratio=2):
    """Run simulations for all parameter combinations and return sorted results."""
    results = []
    total_combinations = len(analysis_levels) * len(threshold_options) * len(risk_options) * len(patience_options) * len(take_profit_options) * len(stop_loss_options)
    with tqdm(total=total_combinations, desc="Running simulations") as pbar:
        for level, threshold, risk_amount, patience, take_profit, stop_loss in product(analysis_levels, threshold_options, risk_options, patience_options, take_profit_options, stop_loss_options):
            define_actions(data, probabilities, threshold, level, True, context_window=None, patience=patience, take_profit=take_profit, stop_loss=stop_loss, reverse=reverse)
            len_diff = 0
            if use_analyzer:
                define_actions_extended(data, cemented_data=cemented_data)
            
            data, capital = run_simulation(data, initial_capital, reverse, risk_amount, expected_profit=expected_profit, risk_percentage=stop_loss, stop_loss_pips=stop_loss_pips, take_profit_pips=take_profit_pips, pip_value=0.0001,
                                           target_rr_ratio=target_rr_ratio)
            debt_profit = data['Profit'].sum()
            results.append({
                "level": level,
                "threshold": threshold,
                "risk_amount": risk_amount,
                "final_capital": capital,
                "debt_profit": debt_profit,
                "patience": patience,
                "take_profit": take_profit,
                "stop_loss": stop_loss
            })
            pbar.update(1)
            
            # if has_consecutive_holds(data['Action'].tolist()):
            #     return results
    
        return sorted(results, key=lambda x: x["debt_profit"], reverse=True)

def process_simulation_window(data, probabilities, analysis_levels, threshold_options, risk_options,
                              patience_options, take_profit_options, stop_loss_options,
                              initial_capital, reverse, expected_profit, output_file, file_path, cemented_data=None, use_analyzer=True, real_time=True, stop_loss_pips=5, take_profit_pips=5, pip_value=0.0001,
                              target_rr_ratio=2):
    """Process a single simulation window."""
    sorted_results = None
    if not real_time:
        sorted_results = process_simulation(data, probabilities, analysis_levels, threshold_options, risk_options, patience_options, take_profit_options, stop_loss_options,
                                            initial_capital, reverse, expected_profit, context_window=None, cemented_data=cemented_data, use_analyzer=use_analyzer, stop_loss_pips=stop_loss_pips, take_profit_pips=take_profit_pips, pip_value=0.0001,
                                            target_rr_ratio=target_rr_ratio)
    else:
        sorted_results = []
        sorted_results.append({
                "level": analysis_levels[0],
                "threshold": threshold_options[0],
                "risk_amount": risk_options[0],
                "final_capital": 0,
                "debt_profit": 0,
                "patience": patience_options[0],
                "take_profit": take_profit_options[0],
                "stop_loss": stop_loss_options[0]
            })
    
    # print_results(sorted_results)
    best_results = sorted_results[:3]
    result = best_results[0]

    predictions = None
    probabilities = None
    if result['threshold'] != 0:
        predictions, probabilities = obtain_final_predictions(file_path)
        predictions, probabilities = pad_predictions_and_probabilities(predictions, probabilities, len(data))

    define_actions(data, probabilities, result['threshold'], result['level'], True, context_window=None, patience=result['patience'], take_profit=result['take_profit'],
                   stop_loss=result['stop_loss'], reverse=reverse)

    if use_analyzer:
        define_actions_extended(data, cemented_data=cemented_data)
    
    if reverse:
        for index, row in data.iterrows():
            if row['Action'] == 'Buy':
                data.loc[index, 'Action'] = 'Sell Short'
            elif row['Action'] == 'Sell Short':
                data.loc[index, 'Action'] = 'Buy'
            elif row['Action'] == 'Sell':
                data.loc[index, 'Action'] = 'Buy to Cover'
            elif row['Action'] == 'Buy to Cover':
                data.loc[index, 'Action'] = 'Sell'
        
    data, _ = run_simulation(data, initial_capital, reverse, result['risk_amount'], expected_profit=None, log=False, risk_percentage=result['stop_loss'], stop_loss_pips=stop_loss_pips, take_profit_pips=take_profit_pips, pip_value=0.0001,
                             target_rr_ratio=target_rr_ratio)
    data['Profit'] = data['Profit'].fillna(0)
    # print_actions(data)
    #   write_to_file(output_file, best_results)
    return data, predictions, probabilities