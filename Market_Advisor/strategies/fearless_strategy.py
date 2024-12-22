from analyzers.indicators import calculate_indicators
from data_helpers.misc import (
	load_data, pad_predictions_and_probabilities,
	calculate_profit_summatory, wins_and_losses_count, pad_take_profits_and_stop_losses,
	cement, check_not_repeated_last_action, print_colored_sentence,
	check_for_expired_actions, check_for_patience,send_action_to_mt5, check_position_by_magic, handle_trailing_stop_loss
)
from analyzers.executer import run_simulation
from analyzers.simulator import process_simulation_window
import time
import MetaTrader5 as mt5
LOG_FILE = "temp_files/last_action_log.txt"

def main_logic(file_path, plotter, storage, reverse=False, multiply_factor=None, expected_profit=1, context_window=100, initial_capital=50000, u_a=True, deep_analysis=False, rt=True,
			   bot_type=7, freeze=20, forced_patience=False, stop_loss_pips=5, take_profit_pips=5, pip_value=0.0001, target_rr_ratio=2, code_multiplier=1, log_file="temp_files/last_action_log.txt", trailing_stop_loss=True, advisors_threshold=3, web_scraping="5m"):
	"""Run the trading simulation."""
	start_time = time.time()
	
	target_rr_ratio = take_profit_pips / stop_loss_pips
	
	if not mt5.initialize():
		print("Failed to initialize MetaTrader5")
		quit()
		
	symbol = "EURUSD"
	if not mt5.symbol_select(symbol, True):
		print(f"Failed to select {symbol}")
		mt5.shutdown()
		quit()
	
	symbol_info = mt5.symbol_info(symbol)

	if symbol_info is None:
		print(f"Failed to get symbol info for {symbol}")

	if not mt5.symbol_select(symbol, True):
		print(f"Failed to select symbol: {symbol}")

	data, predictions, probabilities, analysis_levels, threshold_options, risk_options, patience_options, take_profit_options, stop_loss_options, initial_capital, output_file ,summatory_of_indicators = load_data(file_path=file_path, reverse=reverse, multiply_factor=multiply_factor, initial_capital=initial_capital, deep_calculation=deep_analysis, bot_type=bot_type,web_scraping=web_scraping, advisor_threshold=advisors_threshold)
	
	if len(data) < 20:
		print_colored_sentence("Not enough data to start simulation, need at least 20 rows.")
		return

	c_d = storage.get_dataframe() if storage.get_dataframe() is not None else None
	cement(data, c_d)
	calculate_indicators(data, context_window=context_window)
	data, predictions, probabilities = process_simulation_window(
		data, probabilities, analysis_levels, threshold_options, risk_options,
		patience_options, take_profit_options, stop_loss_options,
		initial_capital, reverse, expected_profit, output_file, file_path,
		cemented_data=c_d, use_analyzer=u_a, real_time=rt, stop_loss_pips=stop_loss_pips, take_profit_pips=take_profit_pips, pip_value=0.0001, target_rr_ratio=target_rr_ratio
	)

	if predictions is not None and probabilities is not None:
		predictions, probabilities = pad_predictions_and_probabilities(predictions, probabilities, len(data))
		
	# print_colored_sentence("Last Action: " + data["Action"].iloc[-1])
	if len(data) < freeze:
		data.loc[:, 'Action'] = 'Hold'
	
	cement(data, c_d)
	check_for_expired_actions(data, patience_options[0])
	if forced_patience:
		check_for_patience(data, patience_options[0])
	check_not_repeated_last_action(data)
	if trailing_stop_loss:
		handle_trailing_stop_loss(data, code_multiplier=code_multiplier)
  
	print_colored_sentence(f"Summatory of Indicators: {summatory_of_indicators}")
	if summatory_of_indicators >= advisors_threshold:
		data.loc[len(data) - 1, 'Action'] = "Buy"
	elif summatory_of_indicators <= -advisors_threshold:
		data.loc[len(data) - 1, 'Action'] = 'Sell Short'
	else:
		if not open(log_file).read():
			data.loc[:len(data)-2, 'Action'] = 'Hold'
	
	check_position_by_magic(str(int("111111") * code_multiplier), log_file=log_file)
	check_position_by_magic(str(int("111222") * code_multiplier), log_file=log_file)
	send_action_to_mt5(data, symbol, stop_loss_pips=stop_loss_pips, take_profit_pips=take_profit_pips, code_multiplier=code_multiplier,
				log_file=log_file)

	data, _ = run_simulation(data, initial_capital, reverse, risk_amount=risk_options[0], expected_profit=None, log=False, risk_percentage=stop_loss_options[0], stop_loss_pips=stop_loss_pips, take_profit_pips=take_profit_pips, pip_value=0.0001,
							 target_rr_ratio=target_rr_ratio)
	calculate_profit_summatory(data)
	pad_take_profits_and_stop_losses(data)
	wins_and_losses_count(data)
	storage.remove_dataframe()
	storage.add_dataframe(data)

	print(f"Time elapsed: {time.time() - start_time}")

	plotter.data_backtest = storage.get_dataframe()
	plotter.predictions = predictions
	plotter.title_suffix = "Real-Time Simulation"
	plotter.update_figure(100)
