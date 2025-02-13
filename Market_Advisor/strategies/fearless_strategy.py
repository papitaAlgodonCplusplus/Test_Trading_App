from analyzers.indicators import calculate_indicators
from data_helpers.misc import (
	load_data, pad_predictions_and_probabilities,
	calculate_profit_summatory, wins_and_losses_count, pad_take_profits_and_stop_losses,
	cement, check_not_repeated_last_action, print_colored_sentence,
	check_for_expired_actions, check_for_patience,send_action_to_mt5, check_position_by_magic
)
from analyzers.executer import run_simulation
from analyzers.simulator import process_simulation_window
import time
import MetaTrader5 as mt5
import pandas as pd

def analyze_trend(data, last_action):
        df = pd.DataFrame(data, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        df['Close'] = df['Close'].astype(float)
        if len(df) < 5:
            return False  
        last_price = df['Close'].iloc[-1]
        if last_price < df['Close'].iloc[-6] and last_action == "Buy":
            return True  
        elif last_price > df['Close'].iloc[-6] and last_action == "Sell Short":
             return True
        return False
    
def main_logic(file_path, plotter, storage, reverse=False, multiply_factor=None, expected_profit=1, context_window=100, initial_capital=50000, u_a=True, deep_analysis=False, rt=True,
			   bot_type=7, freeze=20, forced_patience=False, stop_loss_pips=5, take_profit_pips=5, pip_value=0.0001, target_rr_ratio=2, code_multiplier=1, log_file="temp_files/log.json", trailing_stop_loss=True, advisors_threshold=3, web_scraping="5m"):
	"""Run the trading simulation."""
	start_time = time.time()
	target_rr_ratio = take_profit_pips / stop_loss_pips
	if not mt5.initialize():
		print("Failed to initialize MetaTrader5")
	symbol = "EURUSD"
	if not mt5.symbol_select(symbol, True):
		print(f"Failed to select {symbol}")
		mt5.shutdown()
	symbol_info = mt5.symbol_info(symbol)
	if symbol_info is None:
		print(f"Failed to get symbol info for {symbol}")
	if not mt5.symbol_select(symbol, True):
		print(f"Failed to select symbol: {symbol}")
	data, predictions, probabilities, analysis_levels, threshold_options, risk_options, patience_options, take_profit_options, stop_loss_options, initial_capital, output_file ,summatory_of_indicators = load_data(file_path=file_path, reverse=reverse, multiply_factor=multiply_factor, initial_capital=initial_capital, deep_calculation=deep_analysis, bot_type=bot_type,web_scraping=web_scraping, advisor_threshold=advisors_threshold)
	if len(data) < 20:
		print_colored_sentence("Not enough data to start simulation, need at least 20 rows.")
		return
	print_colored_sentence(f"Summatory of Indicators: {summatory_of_indicators}")
	if summatory_of_indicators >= advisors_threshold:
		data.loc[len(data) - 1, 'Action'] = "Buy"
	elif summatory_of_indicators <= -advisors_threshold:
		data.loc[len(data) - 1, 'Action'] = 'Sell Short'
	else:
		if not open(log_file).read():
			data.loc[:len(data)-2, 'Action'] = 'Hold'
	check_position_by_magic(str(int("111111") * code_multiplier), storage=storage, data=data, 	stop_loss_pips=stop_loss_pips, multiplier=code_multiplier)
	check_position_by_magic(str(int("111222") * code_multiplier), storage=storage, data=data,stop_loss_pips=stop_loss_pips, multiplier=code_multiplier)
	should_reverse = analyze_trend(data, data["Action"].iloc[-1]) 
	print_colored_sentence(f"Should Reverse: {should_reverse}")
	send_action_to_mt5(data, symbol, stop_loss_pips=stop_loss_pips, take_profit_pips=take_profit_pips, code_multiplier=code_multiplier, storage=storage,
                    reverse=False)
	storage.remove_dataframe()
	storage.add_dataframe(data)
	print(f"Time elapsed: {time.time() - start_time}")
