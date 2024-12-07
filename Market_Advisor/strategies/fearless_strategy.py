from analyzers.indicators import calculate_indicators
from data_helpers.misc import (
    load_data, pad_predictions_and_probabilities, update_simulation_data,
    pad_vault, calculate_profit_summatory, wins_and_losses_count, pad_take_profits_and_stop_losses,
    cement, check_not_repeated_last_action, print_colored_sentence,
    check_for_expired_actions
)
from analyzers.executer import run_simulation
from analyzers.simulator import process_simulation_window
import time
    
def main_logic(file_path, plotter, storage, reverse=False, multiply_factor=None, expected_profit=1, context_window=30, initial_capital=50000, u_a=True, deep_analysis=False, rt=True,
               bot_type=1):
    """Run the trading simulation."""
    start_time = time.time()
    data, predictions, probabilities, analysis_levels, threshold_options, risk_options, patience_options, take_profit_options, stop_loss_options, initial_capital, output_file = load_data(file_path=file_path, reverse=reverse, multiply_factor=multiply_factor, initial_capital=initial_capital, deep_calculation=deep_analysis, bot_type=bot_type)

    c_d = storage.get_dataframe() if storage.get_dataframe() is not None else None
    cement(data, c_d)
    if context_window is not None:
        for i in range(context_window, len(data), context_window):
            data_for_simulation = data.loc[i - context_window:i].copy()
            calculate_indicators(data_for_simulation, context_window=None)
            print_colored_sentence(f"Processing window {i - context_window} to {i}")
            data_for_simulation, predictions, probabilities = process_simulation_window(
                data_for_simulation, probabilities, analysis_levels, threshold_options, risk_options,
                patience_options, take_profit_options, stop_loss_options,
                initial_capital, reverse, expected_profit, output_file, file_path,
                cemented_data=c_d, use_analyzer=u_a, real_time=rt
            )
            update_simulation_data(data, data_for_simulation)
    else:
        calculate_indicators(data)
        data, predictions, probabilities = process_simulation_window(
            data, probabilities, analysis_levels, threshold_options, risk_options,
            patience_options, take_profit_options, stop_loss_options,
            initial_capital, reverse, expected_profit, output_file, file_path,
            cemented_data=c_d, use_analyzer=u_a, real_time=rt
        )

    if predictions is not None and probabilities is not None:
        predictions, probabilities = pad_predictions_and_probabilities(predictions, probabilities, len(data))
        
    print_colored_sentence("Last Action: " + data["Action"].iloc[-1])
    cement(data, c_d)
    check_for_expired_actions(data, patience_options[0])
    check_not_repeated_last_action(data)
    pad_vault(data)
    calculate_profit_summatory(data)
    data, _ = run_simulation(data, initial_capital, reverse, risk_amount=risk_options[0], expected_profit=None, log=False, risk_percentage=stop_loss_options[0])
    pad_take_profits_and_stop_losses(data)
    wins_and_losses_count(data)
    storage.remove_dataframe()
    storage.add_dataframe(data)
    
    print(f"Time elapsed: {time.time() - start_time}")
    
    plotter.data_backtest = storage.get_dataframe()
    plotter.predictions = predictions
    plotter.title_suffix = "Real-Time Simulation"
    plotter.update_figure(100)
