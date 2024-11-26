import numpy as np

def get_order_blocks(stock_prices, t, lookback_period=10):
    """
    Identify order blocks (demand or supply zones) based on historical price levels.
    """
    if t < lookback_period:
        return {"demand_zone": False, "supply_zone": False}

    recent_prices = stock_prices[max(0, t-lookback_period):t+1]
    mean_price = np.mean(recent_prices)

    demand_zone = stock_prices[t] < mean_price
    supply_zone = stock_prices[t] > mean_price

    return {"demand_zone": demand_zone, "supply_zone": supply_zone}

def get_poi(stock_prices, t, threshold_factor=1.0):
    """
    Identify points of interest based on sudden price changes relative to historical volatility.
    """
    if t == 0:
        return {"zone": "neutral"}

    price_change = abs(stock_prices[t] - stock_prices[t-1])
    threshold = threshold_factor * np.std(stock_prices[:t+1])  # Standard deviation for threshold

    if price_change > threshold:
        zone = "demand" if stock_prices[t] < stock_prices[t-1] else "supply"
    else:
        zone = "neutral"

    return {"zone": zone}

def get_liquidity_zones(stock_prices, t, consolidation_period=5, threshold_factor=0.01):
    """
    Analyze liquidity zones by identifying price consolidations.
    """
    if t < consolidation_period:
        return {"bullish_liquidity": False, "bearish_liquidity": False}

    recent_prices = stock_prices[t-consolidation_period:t+1]
    price_range = max(recent_prices) - min(recent_prices)
    threshold = threshold_factor * np.mean(stock_prices)  # Relative to the average price

    if price_range < threshold:
        midpoint = (max(recent_prices) + min(recent_prices)) / 2
        bullish_liquidity = stock_prices[t] < midpoint
        bearish_liquidity = stock_prices[t] > midpoint
        return {"bullish_liquidity": bullish_liquidity, "bearish_liquidity": bearish_liquidity}

    return {"bullish_liquidity": False, "bearish_liquidity": False}

def get_divergence_signal(stock_prices, t, volume=None):
    """
    Check for divergence between price movement and volume trends.
    """
    if t < 2:
        return False

    if volume is None:
        volume = np.random.uniform(0.5, 1.5, size=len(stock_prices))  # Mock volume if none provided

    price_trend = stock_prices[t] - stock_prices[t-2]
    volume_trend = volume[t] - volume[t-2]

    # Divergence occurs when price and volume trends are in opposite directions
    return (price_trend > 0 and volume_trend < 0) or (price_trend < 0 and volume_trend > 0)

def get_bos(stock_prices, t):
    """
    Identify Break of Structure (BoS) by observing significant price movements.
    """
    if t < 2:
        return False

    current_price = stock_prices[t]
    previous_prices = stock_prices[t-2:t]
    return current_price > max(previous_prices) or current_price < min(previous_prices)

def get_choc(stock_prices, t):
    """
    Identify Change of Character (ChoC) indicating shifts in market control.
    """
    if t < 3:
        return False

    current_high = max(stock_prices[t], stock_prices[t-1])
    current_low = min(stock_prices[t], stock_prices[t-1])
    return (current_high > max(stock_prices[t-2:t-3])) or (current_low < min(stock_prices[t-2:t-3]))

def get_trend(stock_prices, t, trend_period=3):
    """
    Determine the trend direction (uptrend, downtrend, or neutral) based on recent prices.
    """
    if t < trend_period:
        return "neutral"

    recent_prices = stock_prices[t-trend_period:t+1]
    if recent_prices[-1] > recent_prices[0]:
        return "bullish"
    elif recent_prices[-1] < recent_prices[0]:
        return "bearish"
    else:
        return "neutral"

def get_liquidity_sweep(stock_prices, t, key_levels, margin=0.01):
    """
    Check for liquidity sweeps at predefined key levels.
    """
    if t < 1:
        return False

    price = stock_prices[t]
    return any(abs(price - level) <= margin * level for level in key_levels)
