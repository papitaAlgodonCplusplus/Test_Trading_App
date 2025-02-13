import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from datetime import datetime
from callers.no_strategy_indicators import indicate
def forex_indicate(file_path, web_scraping="5m"):
    version = None if web_scraping == "5m" else "30m" if web_scraping == "30m" else "1h" if web_scraping == "1h" else None
    adviser_recommendation, tcn_recommendation, web_scraper_recommendations, tft_recommendation, lstm_recommendation, dbse_recommendation, garch_recommendation, resnls_recommendation, nbeats_recommendation = indicate(file_path, web_scraping, version=version)
    print(f"Adviser Recommendation: {adviser_recommendation}, TCN Recommendation: {tcn_recommendation}, TFT Recommendation: {tft_recommendation}, LSTM Recommendation: {lstm_recommendation}", f"DBSE Recommendation: {dbse_recommendation}, GARCH Recommendation: {garch_recommendation}", f"ResNLS Recommendation: {resnls_recommendation}, NBEATS Recommendation: {nbeats_recommendation}")
    summatory_of_indicators = adviser_recommendation + tcn_recommendation + tft_recommendation +  lstm_recommendation + dbse_recommendation + garch_recommendation + resnls_recommendation + nbeats_recommendation
    for i, recommendation in enumerate(web_scraper_recommendations):
        print(f"Web Scraper Recommendation {i + 1}: {recommendation}")
        summatory_of_indicators += recommendation
    return summatory_of_indicators