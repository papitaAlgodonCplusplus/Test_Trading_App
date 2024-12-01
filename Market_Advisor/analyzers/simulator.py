
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

        profit_over_time.append(capital - initial_capital)
        if log:
           print(f"Date: {row['Date']}, Capital: {capital:.2f}, Units: {units_hold:.2f}, Profit: {profit_over_time[-1]:.2f}, Action: {action}, Price: {price:.2f}")

        if profit_over_time[-1] > 0:
            profit_vault += profit_over_time[-1]
            capital = initial_capital

        vault_values.append(profit_vault)    

        if expected_profit and profit_vault >= expected_profit:
            capital += units_hold * price
            units_hold = 0
            return data, capital, profit_vault

    if units_hold > 0:
        capital += units_hold * data['Close'].iloc[-1]
        profit_over_time[-1] = capital - initial_capital

    data.loc[:, 'Profit'] = profit_over_time
    data.loc[:, 'Vault'] = vault_values
    return data, capital, profit_vault
