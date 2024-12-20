from ta.momentum import RSIIndicator

def calculate_indicators(data, context_window=None):
    """Calculate technical indicators."""
    data_backup = data.copy()

    if context_window is not None:
        try:
            data_backup = data_backup.iloc[:context_window].copy()
        except Exception as e:
            print(f"Error occurred while calculating indicators: {e}")

    data_backup['SMA_20'] = data_backup['Close'].rolling(window=20).mean()
    data_backup['EMA_12'] = data_backup['Close'].ewm(span=12, adjust=False).mean()
    data_backup['EMA_26'] = data_backup['Close'].ewm(span=26, adjust=False).mean()
    data_backup['MACD'] = data_backup['EMA_12'] - data_backup['EMA_26']
    data_backup['Signal_Line'] = data_backup['MACD'].ewm(span=9, adjust=False).mean()
    data_backup['RSI'] = RSIIndicator(data_backup['Close'], window=14).rsi()
    data_backup['High_Liquidity'] = data_backup['High'].rolling(window=5).max()
    data_backup['Low_Liquidity'] = data_backup['Low'].rolling(window=5).min()

    data_backup['Prev_Close'] = data_backup['Close'].shift(1)
    data_backup['True_Range'] = data_backup[['High', 'Low', 'Prev_Close']].apply(
        lambda row: max(
            row['High'] - row['Low'],
            abs(row['High'] - row['Prev_Close']),
            abs(row['Low'] - row['Prev_Close'])
        ),
        axis=1
    )

    rolling_window = 14
    rolling_min = data_backup['True_Range'].rolling(window=rolling_window).min()
    rolling_max = data_backup['True_Range'].rolling(window=rolling_window).max()
    data_backup['Liquidity_Pool'] = (data_backup['True_Range'] - rolling_min) / (rolling_max - rolling_min)
    data_backup['Liquidity_Pool'] = data_backup['Liquidity_Pool'].fillna(0)

    # Bollinger Bands
    data_backup['BB_Upper'] = data_backup['SMA_20'] + 2 * data_backup['Close'].rolling(window=20).std()
    data_backup['BB_Lower'] = data_backup['SMA_20'] - 2 * data_backup['Close'].rolling(window=20).std()

    # Average True Range
    data_backup['ATR'] = data_backup['True_Range'].rolling(window=rolling_window).mean()

    # Order Block Validity
    data_backup['Order_Block_Valid'] = (
        (data_backup['Low'].shift(1) > data_backup['Low']) & (data_backup['High'].shift(-1) < data_backup['Low'])
    )

    for column in [
        'SMA_20', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line', 'RSI',
        'High_Liquidity', 'Low_Liquidity', 'True_Range', 'Liquidity_Pool',
        'BB_Upper', 'BB_Lower', 'ATR', 'Order_Block_Valid'
    ]:
        data.loc[data_backup.index, column] = data_backup[column]

    data.loc[:, 'Action'] = 'Hold'
    return data

def get_conditions(data, level):
    """Define bullish and bearish conditions based on analysis level."""
    bullish = (data['MACD'] > data['Signal_Line'])
    bearish = (data['MACD'] < data['Signal_Line'])

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
        volatility_threshold = data['ATR'].mean() * 0.5  # Example: 50% of the mean ATR
        bullish &= (data['ATR'] > volatility_threshold) & (data['ATR'] < 2 * volatility_threshold)
        bearish &= (data['ATR'] > volatility_threshold) & (data['ATR'] < 2 * volatility_threshold)

    return bullish, bearish