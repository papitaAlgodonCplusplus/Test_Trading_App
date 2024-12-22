import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from callers.no_strategy_indicators import indicate
from strategies.fibonacci import fibonacci_calculator
from strategies.price_range import price_range_calculator
from strategies.stop_hunt import stop_hunt_calculator

def forex_indicate(file_path, web_scraping="5m"):
    version = None if web_scraping == "5m" else "30m" if web_scraping == "30m" else "1h" if web_scraping == "1h" else None
    adviser_recommendation, tcn_recommendation, web_scraper_recommendations, tft_recommendation = indicate(file_path, web_scraping, version=version)
    print(f"Adviser Recommendation: {adviser_recommendation}, TCN Recommendation: {tcn_recommendation}, TFT Recommendation: {tft_recommendation}")
    # TODO: Check on the windows of these strategies
    trade_signal, date = fibonacci_calculator(file_path)
    print(f"Fibonacci Trade Signal: {trade_signal}, Date: {date}")
    trade_signal_ranged, date_ranged = price_range_calculator(file_path)
    print(f"Price Range Trade Signal: {trade_signal_ranged}, Date: {date_ranged}")
    trade_signal_stop_hunt, date_stop_hunt = stop_hunt_calculator(file_path)
    print(f"Stop Hunt Trade Signal: {trade_signal_stop_hunt}, Date: {date_stop_hunt}")
    summatory_of_indicators = adviser_recommendation + tcn_recommendation + tft_recommendation +    trade_signal + trade_signal_ranged + trade_signal_stop_hunt
    for i, recommendation in enumerate(web_scraper_recommendations):
        print(f"Web Scraper Recommendation {i + 1}: {recommendation}")
        summatory_of_indicators += recommendation
    return summatory_of_indicators