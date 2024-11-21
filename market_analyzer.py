import numpy as np

def get_order_blocks(stock_prices, t, lookback_period=10):
    """
    Identify order blocks (demand or supply zones).
    """
    demand_zone = stock_prices[t] < np.mean(stock_prices[max(0, t-lookback_period):t+1])
    supply_zone = stock_prices[t] > np.mean(stock_prices[max(0, t-lookback_period):t+1])
    
    return {"demand_zone": demand_zone, "supply_zone": supply_zone}

def get_poi(stock_prices, t, threshold_factor=1.0):
    """
    Identify points of interest based on sudden price changes.
    """
    if t == 0:
        return {"zone": "neutral"}

    price_change = abs(stock_prices[t] - stock_prices[t-1])
    threshold = threshold_factor * np.std(stock_prices[:t+1])  # Using standard deviation as the threshold
    if price_change > threshold:
        zone = "demand" if stock_prices[t] < stock_prices[t-1] else "supply"
    else:
        zone = "neutral"

    return {"zone": zone}

def get_liquidity_zones(stock_prices, t, consolidation_period=5, threshold_factor=0.01):
    """
    Analyze liquidity zones by looking for consolidation periods.
    """
    if t < consolidation_period:  # Ensure enough data points to analyze
        return False

    recent_prices = stock_prices[t-consolidation_period:t+1]
    price_range = max(recent_prices) - min(recent_prices)
    threshold = threshold_factor * np.mean(stock_prices)  # Mock: 1% of the average price
    return price_range < threshold

def get_divergence_signal(stock_prices, t, volume=None):
    """
    Check for divergence in price and a mock volume indicator.
    """
    if t < 2:  # Ensure we have enough data points
        return False

    if volume is None:
        volume = np.random.uniform(0.5, 1.5, size=len(stock_prices))

    price_trend = stock_prices[t] - stock_prices[t-2]
    volume_trend = volume[t] - volume[t-2]
    return (price_trend > 0 and volume_trend < 0) or (price_trend < 0 and volume_trend > 0)


def get_bos(stock_prices, t):
    """Identify Break of Structure (BoS) by checking solid body closes."""
    if t < 2:
        return False
    
    body_close = stock_prices[t]
    previous_body = stock_prices[t-1]
    return body_close > max(previous_body, stock_prices[t-2]) or body_close < min(previous_body, stock_prices[t-2])

def get_choc(stock_prices, t):
    """Identify Change of Character (ChoC) by analyzing shifts in control."""
    if t < 2:
        return False

    current_high = max(stock_prices[t], stock_prices[t-1])
    current_low = min(stock_prices[t], stock_prices[t-1])
    return current_high > max(stock_prices[t-2], stock_prices[t-3]) or current_low < min(stock_prices[t-2], stock_prices[t-3])

def get_trend(stock_prices, t, trend_period=3):
    """Determine the trend direction (uptrend, downtrend, or neutral)."""
    if t < trend_period:
        return "neutral"

    recent_prices = stock_prices[t-trend_period:t+1]
    if recent_prices[-1] > recent_prices[0]:
        return "uptrend"
    elif recent_prices[-1] < recent_prices[0]:
        return "downtrend"
    else:
        return "neutral"

def get_liquidity_sweep(stock_prices, t, key_levels):
    """Check for liquidity sweeps at key levels."""
    if t < 1:
        return False

    price = stock_prices[t]
    swept = any(abs(price - level) < 0.01 * level for level in key_levels)  # 1% margin
    return swept
