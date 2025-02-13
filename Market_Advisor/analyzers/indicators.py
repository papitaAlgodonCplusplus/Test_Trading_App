from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import pandas as pd
import numpy as np
import random

def calculate_indicators(data, context_window=100):
    if len(data) < context_window:
        context_window = len(data)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Trend'] = ''
    data['Support'] = np.nan
    data['Resistance'] = np.nan
    data['SMA'] = np.nan
    data['EMA'] = np.nan
    data['Order_Block'] = ''
    data['Candle_Pattern'] = ''
    data['BB_High'] = np.nan
    data['BB_Low'] = np.nan
    data['MACD'] = np.nan
    data['MACD_Signal'] = np.nan
    data['Stoch_%K'] = np.nan
    data['Stoch_%D'] = np.nan
    data['OBV'] = np.nan
    data['Z-Score'] = np.nan
    data['ATR'] = np.nan
    data['Doji'] = False
    rsi = RSIIndicator(close=data['Close'])
    data['RSI'] = rsi.rsi()
    bb = BollingerBands(close=data['Close'], window=context_window)
    macd = MACD(close=data['Close'])
    stoch = StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'])
    atr = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=min(14, context_window))
    obv = OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume'])
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    for i in range(len(data)):
        window_data = data.iloc[:i+1]
        if i > 0:
            if window_data['Close'].iloc[-1] > window_data['Close'].iloc[-2]:
                data.at[i, 'Trend'] = 'Bullish'
            elif window_data['Close'].iloc[-1] < window_data['Close'].iloc[-2]:
                data.at[i, 'Trend'] = 'Bearish'
        if i >= context_window - 1:
            data.at[i, 'SMA'] = window_data['Close'].iloc[-context_window:].mean()
            data.at[i, 'EMA'] = window_data['Close'].iloc[-context_window:].ewm(span=context_window, adjust=False).mean().iloc[-1]
        if i >= context_window - 1:
            data.at[i, 'Support'] = window_data['Low'].iloc[-context_window:].min()
            data.at[i, 'Resistance'] = window_data['High'].iloc[-context_window:].max()
        if i > 0 and (window_data['Low'].iloc[-1] < data.at[i-1, 'Support']):
            data.at[i, 'Order_Block'] = 'Potential Bullish Order Block'
        elif i > 0 and (window_data['High'].iloc[-1] > data.at[i-1, 'Resistance']):
            data.at[i, 'Order_Block'] = 'Potential Bearish Order Block'
        if i >= 14:
            avg_gain = gain.iloc[:i+1].rolling(window=14).mean().iloc[-1]
            avg_loss = loss.iloc[:i+1].rolling(window=14).mean().iloc[-1]
            if avg_loss == 0:
                data.at[i, 'RSI'] = 100
            else:
                rs = avg_gain / avg_loss
                data.at[i, 'RSI'] = 100 - (100 / (1 + rs))
        if abs(window_data['Close'].iloc[-1] - window_data['Open'].iloc[-1]) <= (window_data['High'].iloc[-1] - window_data['Low'].iloc[-1]) * 0.1:
            data.at[i, 'Doji'] = True
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['Stoch_%K'] = stoch.stoch()
    data['Stoch_%D'] = stoch.stoch_signal()
    data['OBV'] = obv.on_balance_volume()
    data['Z-Score'] = (data['Close'] - data['Close'].rolling(context_window).mean()) / data['Close'].rolling(context_window).std()
    data['ATR'] = atr.average_true_range()
    return data

def get_conditions(data, level, threshold=0.5, reverse=False):
    buy_long = pd.Series(False, index=data.index)
    sell_long = pd.Series(False, index=data.index)
    sell_short = pd.Series(False, index=data.index)
    buy_to_cover = pd.Series(False, index=data.index)
    
    if level == 0:
        buy_long = data['Trend'] == 'Bullish'
        sell_long = data['Trend'] == 'Bearish'
        sell_short = data['Trend'] == 'Bearish'
        buy_to_cover = data['Trend'] == 'Bullish'

    elif level == 1:
        buy_long = data['Trend'] == 'Bullish'
        sell_long = data['Trend'] == 'Bearish'
        sell_short = (data['Trend'] == 'Bearish') & (data['Close'] < data['SMA'])
        buy_to_cover = (data['Trend'] == 'Bullish') & (data['Close'] > data['SMA'])

    elif level == 2:
        buy_long = (data['Close'] > data['Support']) & (data['Close'] > data['SMA'])
        sell_long = (data['Close'] < data['Resistance']) & (data['Close'] < data['SMA'])
        sell_short = (data['Trend'] == 'Bearish') | (data['High'] < data['Resistance'])
        buy_to_cover = (data['Trend'] == 'Bullish') | (data['Low'] > data['Support'])

    elif level == 3:
        buy_long = (data['Close'] > data['EMA']) & (data['RSI'] < 65) & (data['Low'] > data['Support'])
        sell_long = (data['Close'] < data['EMA']) & (data['RSI'] > 35) & (data['High'] < data['Resistance'])
        sell_short = (data['Close'] < data['EMA']) & (data['RSI'] > 60)
        buy_to_cover = (data['Close'] > data['EMA']) & (data['RSI'] < 40)

    elif level == 4:
        buy_long = (data['Candle_Pattern'] == 'Bullish Engulfing') | (data['Close'] > data['EMA'])
        sell_long = (data['Candle_Pattern'] == 'Bearish Engulfing') | (data['Close'] < data['EMA'])
        sell_short = (data['Candle_Pattern'] == 'Bullish Engulfing') | (data['RSI'] > 70)
        buy_to_cover = (data['Candle_Pattern'] == 'Bearish Engulfing') | (data['RSI'] < 30)

    elif level == 5:
        buy_long = (data['Stoch_%K'] < 20) & (data['Low'] > data['Support'])
        sell_long =  (data['Stoch_%K'] > 80) & (data['High'] < data['Resistance'])
        sell_short = (data['Close'] > data['BB_High']) | (data['RSI'] > 70)
        buy_to_cover = (data['Close'] < data['BB_Low']) | (data['RSI'] < 30)

    elif level == 6:
        buy_long = (data['Stoch_%K'] < 20) | (data['RSI'] < 35)
        sell_long = (data['Stoch_%K'] > 80) | (data['RSI'] > 65)
        sell_short = (data['MACD'] < data['MACD_Signal']) | (data['Close'] < data['SMA'])
        buy_to_cover = (data['MACD'] > data['MACD_Signal']) | (data['Close'] > data['SMA'])

    if level >= 1:
        bullish &= (data['RSI'] < 70)
        bearish &= (data['RSI'] > 30)
    if level >= 2:
        bullish &= (data['Close'] > data['SMA_20'])
        bearish &= (data['Close'] < data['SMA_20'])
    if level >= 3:
        bullish &= (data['Liquidity_Pool'] > 0.5)
        bearish &= (data['Liquidity_Pool'] < 0.5)
        bullish |= data['Order_Block_Valid'] & (data['Close'] > data['Low'].shift(1))
        bearish |= data['Order_Block_Valid'] & (data['Close'] < data['High'].shift(1))
    if level >= 4:
        bullish &= (data['Close'] <= data['BB_Lower']) & (data['RSI'] < 70)
        bearish &= (data['Close'] >= data['BB_Upper']) & (data['RSI'] > 30)
        volatility_threshold = data['ATR'].mean() * 0.5
        bullish |= (data['ATR'] > volatility_threshold) & (data['ATR'] < 2 * volatility_threshold)
        bearish |= (data['ATR'] > volatility_threshold) & (data['ATR'] < 2 * volatility_threshold)
    if level >= 5:
        bullish |= (data['Close'] <= data['BB_Lower']) & (data['RSI'] < 70)
        bearish |= (data['Close'] >= data['BB_Upper']) & (data['RSI'] > 30)
        volatility_threshold = data['ATR'].mean() * 0.5
        bullish &= (data['ATR'] > volatility_threshold) & (data['ATR'] < 2 * volatility_threshold)
        bearish &= (data['ATR'] > volatility_threshold) & (data['ATR'] < 2 * volatility_threshold)
    elif level == 7:
        buy_long = (data['OBV'] > data['OBV'].rolling(5).mean()) | (data['Close'] > data['SMA']) & (data['ATR'] < data['ATR'].rolling(5).mean())
        sell_long = (data['OBV'] < data['OBV'].rolling(5).mean()) | (data['Close'] < data['SMA']) & (data['ATR'] > data['ATR'].rolling(5).mean())
        sell_short = (data['Z-Score'] > 2) & (data['Close'] < data['EMA'])
        buy_to_cover = (data['Z-Score'] < -2) & (data['Close'] > data['EMA'])

    for i in range(len(data)):
        if buy_long[i] and buy_to_cover[i]:
            if random.random() > threshold:
                buy_to_cover[i] = False
            else:
                buy_long[i] = False
        if sell_long[i] and sell_short[i]:
            if random.random() > threshold:
                sell_short[i] = False
            else:
                sell_long[i] = False
    
    if reverse:
        buy_long, sell_long, sell_short, buy_to_cover = sell_short, buy_to_cover, buy_long, sell_long
                
    long_position_open = False
    short_position_open = False
    for i in range(len(data)):
        if buy_long[i]:
            long_position_open = True
        if sell_long[i] and long_position_open:
            long_position_open = False
        else:
            sell_long[i] = False
        if sell_short[i]:
            short_position_open = True
        if buy_to_cover[i] and short_position_open:
            short_position_open = False
        else:
            buy_to_cover[i] = False
    return buy_long, sell_long, sell_short, buy_to_cover    