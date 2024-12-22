import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from . import adviser
from . import tcn_caller
from . import web_scraper
from . import tft_caller
import MetaTrader5 as mt5
import pandas as pd

def indicate(data_path, web_scraping="5m", version=None):
    if not mt5.initialize():
        print("Failed to initialize MetaTrader5")
        quit()
        
    symbol = "EURUSD"
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")
        mt5.shutdown()
        quit()
    
    len_data = len(pd.read_csv(data_path))
    adviser_recommendation = 0
    tft_recommendation = 0
    if len_data >= 134:
        adviser_recommendation = adviser.indicate(data_path, version)
        tft_recommendation = tft_caller.indicate(data_path, version)
    tcn_recommendation = tcn_caller.indicate(data_path, version)
    web_recommendations = []
    web_recommendations.append(web_scraper.indicate_investing(web_scraping))
    web_recommendations.append(web_scraper.indicate_tradingview(web_scraping))
    web_recommendations.append(web_scraper.indicate_tradersunion(''.join(reversed(web_scraping))))
    return adviser_recommendation, tcn_recommendation, web_recommendations, tft_recommendation