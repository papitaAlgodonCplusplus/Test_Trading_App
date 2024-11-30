from analyzers.indicators import calculate_indicators
from data_helpers.misc import (
    load_data, pad_predictions_and_probabilities, update_simulation_data,
    pad_vault, process_simulation_window
)

def main_logic(file_path, plotter, reverse=False, multiply_factor=None, expected_profit=1, context_window=30, initial_capital=50000):
    """Run the trading simulation."""
    data, predictions, probabilities, results, analysis_levels, threshold_options, risk_options, _, output_file = load_data(file_path, reverse, multiply_factor)

    if context_window is not None:
        for i in range(context_window, len(data), context_window):
            data_for_simulation = data.loc[i - context_window:i].copy()
            calculate_indicators(data_for_simulation, context_window=context_window)
            data_for_simulation, predictions, probabilities = process_simulation_window(
                data_for_simulation, probabilities, analysis_levels, threshold_options, risk_options,
                initial_capital, reverse, expected_profit, context_window, output_file, file_path
            )
            update_simulation_data(data, data_for_simulation)
    else:
        calculate_indicators(data)
        data, predictions, probabilities = process_simulation_window(
            data, probabilities, analysis_levels, threshold_options, risk_options,
            initial_capital, reverse, expected_profit, context_window, output_file, file_path
        )

    predictions, probabilities = pad_predictions_and_probabilities(predictions, probabilities, len(data))
    pad_vault(data)
    plotter.data = data
    plotter.predictions = predictions
    plotter.title_suffix = "Real-Time Simulation"
    plotter.update_figure()
