import numpy as np
from market_analyzer import (
    get_trend, get_order_blocks, get_poi, get_liquidity_zones, get_liquidity_sweep,
    get_bos, get_choc, get_divergence_signal
)

class ActionService:
    # -------------------- ACTIONS --------------------
    @staticmethod
    def hold(agent, stock_prices, t):
        """Wait without taking action."""
        trend_direction = get_trend(stock_prices, t)
        if trend_direction == "neutral":
            return {
                "status": "success",
                "action": "hold",
                "details": {
                    "trend": "neutral",
                    "balance": agent.balance,
                    "inventory": agent.inventory,
                }
            }
        return {
            "status": "success",
            "action": "hold",
            "details": {
                "trend": "other",
                "balance": agent.balance,
                "inventory": agent.inventory,
            }
        }

    @staticmethod
    def buy(agent, stock_prices, t):
        """Execute a buy action if conditions are met."""
        order_blocks = get_order_blocks(stock_prices, t)
        if order_blocks is None:
            order_blocks = {}
        if agent.balance > stock_prices[t] and order_blocks.get('demand_zone', False):
            agent.balance -= stock_prices[t]
            agent.inventory.append(stock_prices[t])
            return {
                "status": "success",
                "action": "buy",
                "details": {"price": stock_prices[t], "balance": agent.balance, "inventory": agent.inventory}
            }
        return {
            "status": "skipped",
            "action": "buy",
            "details": {"reason": "Insufficient Funds or No Demand Zone"}
        }

    @staticmethod
    def sell(agent, stock_prices, t):
        """Execute a sell action if conditions are met."""
        liquidity_zone = get_liquidity_zones(stock_prices, t)
        divergence_signal = get_divergence_signal(stock_prices, t)

        if len(agent.inventory) > 0 and liquidity_zone and divergence_signal:
            agent.balance += stock_prices[t]
            bought_price = agent.inventory.pop(0)
            profit = stock_prices[t] - bought_price
            return {
                "status": "success",
                "action": "sell",
                "details": {"price": stock_prices[t], "profit": profit, "balance": agent.balance, "inventory": agent.inventory}
            }
        return {
            "status": "skipped",
            "action": "sell",
            "details": {"reason": "No Liquidity Zone or Divergence Signal"}
        }

    @staticmethod
    def place_pending_buy(agent, stock_prices, t, key_levels):
        """Place a pending buy order in a liquidity zone."""
        liquidity_sweep = get_liquidity_sweep(stock_prices, t, key_levels)
        if agent.balance > stock_prices[t] and liquidity_sweep:
            agent.balance -= stock_prices[t]  # Reserve funds immediately
            agent.pending_orders.append(("buy", stock_prices[t]))
            return {
                "status": "success",
                "action": "pending_buy",
                "details": {
                    "price": stock_prices[t],
                    "balance": agent.balance,
                    "inventory": agent.inventory,
                }
            }
        return {
            "status": "skipped",
            "action": "pending_buy",
            "details": {
                "reason": "No Liquidity Sweep or Insufficient Funds",
                "balance": agent.balance,
                "inventory": agent.inventory,
            }
        }

    @staticmethod
    def place_pending_sell(agent, stock_prices, t, key_levels):
        """Place a pending sell order in a liquidity zone."""
        liquidity_sweep = get_liquidity_sweep(stock_prices, t, key_levels)
        if len(agent.inventory) > 0 and liquidity_sweep:
            agent.pending_orders.append(("sell", stock_prices[t]))
            return {
                "status": "success",
                "action": "pending_sell",
                "details": {
                    "price": stock_prices[t],
                    "balance": agent.balance,
                    "inventory": agent.inventory,
                }
            }
        return {
            "status": "skipped",
            "action": "pending_sell",
            "details": {
                "reason": "No Liquidity Sweep or Empty Inventory",
                "balance": agent.balance,
                "inventory": agent.inventory,
            }
        }

    @staticmethod
    def execute_pending_orders(agent, stock_prices, t):
        """Check and execute pending orders if conditions are met."""
        executed_orders = []
        for order in agent.pending_orders[:]:
            order_type, target_price = order

            if order_type == "buy" and agent.balance > target_price and stock_prices[t] <= target_price:
                agent.balance -= target_price
                agent.inventory.append(target_price)
                executed_orders.append({"type": "buy", "price": target_price})

            elif order_type == "sell" and len(agent.inventory) > 0 and stock_prices[t] >= target_price:
                bought_price = agent.inventory.pop(0)
                profit = target_price - bought_price
                agent.balance += target_price
                executed_orders.append({"type": "sell", "price": target_price, "profit": profit})

        # Remove executed orders
        agent.pending_orders = [order for order in agent.pending_orders if {"type": order[0], "price": order[1]} not in executed_orders]

        return {
            "status": "executed",
            "action": "execute_pending_orders",
            "details": {"executed_orders": executed_orders, "balance": agent.balance, "inventory": agent.inventory}
        }

    # -------------------- AUXILIARY ANALYSIS FUNCTIONS --------------------
    @staticmethod
    def analyze_trend(stock_prices, t):
        """Returns the market trend direction."""
        return get_trend(stock_prices, t)

    @staticmethod
    def evaluate_order_blocks(stock_prices, t):
        """Checks for demand and supply zones in the market."""
        return get_order_blocks(stock_prices, t)

    @staticmethod
    def analyze_poi(stock_prices, t):
        """Returns points of interest (POI) in the market."""
        return get_poi(stock_prices, t)

    @staticmethod
    def evaluate_liquidity_zones(stock_prices, t):
        """Identifies liquidity zones in the market."""
        return get_liquidity_zones(stock_prices, t)

    @staticmethod
    def analyze_divergence(stock_prices, t):
        """Detects divergence signals for trend reversals."""
        return get_divergence_signal(stock_prices, t)

    @staticmethod
    def evaluate_bos(stock_prices, t):
        """Check for Break of Structure (BoS)."""
        return get_bos(stock_prices, t)

    @staticmethod
    def evaluate_choc(stock_prices, t):
        """Check for Change of Character (ChoC)."""
        return get_choc(stock_prices, t)

    @staticmethod
    def evaluate_liquidity_sweep(stock_prices, t, key_levels):
        """Check for liquidity sweeps at key levels."""
        return get_liquidity_sweep(stock_prices, t, key_levels)
