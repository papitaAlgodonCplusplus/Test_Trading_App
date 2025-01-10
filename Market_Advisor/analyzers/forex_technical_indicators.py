import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from datetime import datetime
from callers.no_strategy_indicators import indicate
from strategies.fibonacci import fibonacci_calculator
from strategies.price_range import price_range_calculator
from strategies.stop_hunt import stop_hunt_calculator

def forex_indicate(file_path, web_scraping="5m"):
    version = None if web_scraping == "5m" else "30m" if web_scraping == "30m" else "1h" if web_scraping == "1h" else None
    adviser_recommendation, tcn_recommendation, web_scraper_recommendations, tft_recommendation, lstm_recommendation, dbse_recommendation, garch_recommendation, resnls_recommendation, nbeats_recommendation = indicate(file_path, web_scraping, version=version)
    print(f"Adviser Recommendation: {adviser_recommendation}, TCN Recommendation: {tcn_recommendation}, TFT Recommendation: {tft_recommendation}, LSTM Recommendation: {lstm_recommendation}", f"DBSE Recommendation: {dbse_recommendation}, GARCH Recommendation: {garch_recommendation}", f"ResNLS Recommendation: {resnls_recommendation}, NBEATS Recommendation: {nbeats_recommendation}")
    trade_signal, date = fibonacci_calculator(file_path)
    now = datetime.now()
    if date is not None:
        date = datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S")
        if (now - date).seconds > 300:
            trade_signal = 0
    trade_signal_ranged, date_ranged = price_range_calculator(file_path)
    if date_ranged is not None:
        date_ranged = datetime.strptime(str(date_ranged), "%Y-%m-%d %H:%M:%S")
        if (now - date_ranged).seconds > 300:
            trade_signal_ranged = 0
    trade_signal_stop_hunt, date_stop_hunt = stop_hunt_calculator(file_path)
    if date_stop_hunt is not None:
        date_stop_hunt = datetime.strptime(str(date_stop_hunt), "%Y-%m-%d %H:%M:%S")
        if (now - date_stop_hunt).seconds > 300:
            trade_signal_stop_hunt = 0
    summatory_of_indicators = adviser_recommendation + tcn_recommendation + tft_recommendation +  lstm_recommendation + trade_signal + trade_signal_ranged + trade_signal_stop_hunt + dbse_recommendation + garch_recommendation + resnls_recommendation + nbeats_recommendation
    for i, recommendation in enumerate(web_scraper_recommendations):
        print(f"Web Scraper Recommendation {i + 1}: {recommendation}")
        summatory_of_indicators += recommendation
    return summatory_of_indicators