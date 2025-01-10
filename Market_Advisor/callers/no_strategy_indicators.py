import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from . import price_predicter_caller
from . import dbse_caller
from . import web_scraper
from . import garch_caller
from . import resnls_caller
from . import nbeats_caller
import MetaTrader5 as mt5
import pandas as pd

def indicate(data_path, web_scraping="5m", version=None):
    if not mt5.initialize():
        print("Failed to initialize MetaTrader5")
        
    symbol = "EURUSD"
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")
        mt5.shutdown()
    
    len_data = len(pd.read_csv(data_path))
    adviser_recommendation = 0
    tft_recommendation = 0
    tcn_recommendation = 0
    lstm_recommendation = 0
    dbse_recommendation = 0
    garch_recommendation = 0
    resnls_recommendation = 0
    nbeats_recommendation = 0
    if len_data >= 35:
        lstm_recommendation = price_predicter_caller.indicate()
        dbse_recommendation = dbse_caller.indicate()
        garch_recommendation = garch_caller.indicate()
        resnls_recommendation = resnls_caller.indicate() * 2
        nbeats_recommendation = nbeats_caller.indicate()
    web_recommendations = []
    # web_recommendations.append(web_scraper.indicate_fxleaders(web_scraping))
    # web_recommendations.append(web_scraper.indicate_investing(web_scraping))
    # web_recommendations.append(web_scraper.indicate_tradingview(web_scraping))
    # web_recommendations.append(web_scraper.indicate_tradersunion(''.join(reversed(web_scraping))))
    web_recommendations.append(web_scraper.indicate_fxstreet())
    return adviser_recommendation, tcn_recommendation, web_recommendations, tft_recommendation, lstm_recommendation, dbse_recommendation, garch_recommendation, resnls_recommendation, nbeats_recommendation