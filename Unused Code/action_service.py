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
        return {
            "status": "success",
            "action": "hold",
            "details": {
                "trend": trend_direction,
                "balance": agent.balance,
                "inventory": agent.inventory,
            }
        }

    @staticmethod
    def buy(agent, stock_prices, t, quantity=1):
        """Execute a buy action immediately if the agent has enough balance for the specified quantity."""
        total_cost = stock_prices[t] * int(quantity)
        if agent.balance >= total_cost:
            agent.balance -= total_cost
            agent.inventory.extend([stock_prices[t]] * int(quantity))
            return {
                "status": "success",
                "action": "buy_instant",
                "details": {
                    "price_per_unit": stock_prices[t],
                    "quantity": quantity,
                    "total_cost": total_cost,
                    "balance": agent.balance,
                    "inventory": agent.inventory
                }
            }
        return {
            "status": "skipped",
            "action": "buy_instant",
            "details": {"reason": "Insufficient Funds", "required": total_cost, "balance": agent.balance}
        }

    @staticmethod
    def buy_when_demand(agent, stock_prices, t, quantity=1):
        """Execute a buy action only if a demand zone condition is met and balance is sufficient."""
        order_blocks = get_order_blocks(stock_prices, t)
        total_cost = stock_prices[t] * int(quantity)
        if agent.balance >= total_cost and order_blocks.get('demand_zone', False):
            agent.balance -= total_cost
            agent.inventory.extend([stock_prices[t]] * int(quantity))
            return {
                "status": "success",
                "action": "buy_when_demand",
                "details": {
                    "price_per_unit": stock_prices[t],
                    "quantity": quantity,
                    "total_cost": total_cost,
                    "balance": agent.balance,
                    "inventory": agent.inventory
                }
            }
        return {
            "status": "skipped",
            "action": "buy_when_demand",
            "details": {"reason": "Insufficient Funds or No Demand Zone"}
        }

    @staticmethod
    def sell_instant(agent, stock_prices, t, _):
        """Sell instantly if inventory is available for the specified quantity."""
        quantity = len(agent.inventory)
        if True:
            sold_items = agent.inventory[:int(quantity)]
            agent.inventory = agent.inventory[int(quantity):]
            total_revenue = stock_prices[t] * int(quantity)
            profit = total_revenue - sum(sold_items[:int(quantity)])
            agent.balance += total_revenue
            return {
                "status": "success",
                "action": "sell_instant",
                "details": {
                    "price_per_unit": stock_prices[t],
                    "quantity": quantity,
                    "total_revenue": total_revenue,
                    "profit": profit,
                    "balance": agent.balance,
                    "inventory": agent.inventory
                }
            }
        return {
            "status": "skipped",
            "action": "sell_instant",
            "details": {"reason": "Not enough inventory", "available_inventory": len(agent.inventory)}
        }

    @staticmethod
    def sell_when_feasible_1(agent, stock_prices, t, _):
        """Sell when liquidity zones indicate high activity."""
        liquidity_zone = get_liquidity_zones(stock_prices, t)
        quantity = len(agent.inventory)
        if liquidity_zone:
            sold_items = agent.inventory[:int(quantity)]
            agent.inventory = agent.inventory[int(quantity):]
            total_revenue = stock_prices[t] * int(quantity)
            profit = total_revenue - sum(sold_items[:int(quantity)])
            agent.balance += total_revenue
            return {
                "status": "success",
                "action": "sell_when_feasible_1",
                "details": {
                    "price_per_unit": stock_prices[t],
                    "quantity": quantity,
                    "total_revenue": total_revenue,
                    "profit": profit,
                    "balance": agent.balance,
                    "inventory": agent.inventory
                }
            }
        return {
            "status": "skipped",
            "action": "sell_when_feasible_1",
            "details": {"reason": "No inventory or liquidity zone"}
        }

    @staticmethod
    def sell_when_feasible_2(agent, stock_prices, t, _):
        """Sell when divergence signals indicate an opportunity."""
        divergence_signal = get_divergence_signal(stock_prices, t)
        quantity = len(agent.inventory)
        if divergence_signal:
            sold_items = agent.inventory[:int(quantity)]
            agent.inventory = agent.inventory[int(quantity):]
            total_revenue = stock_prices[t] * int(quantity)
            profit = total_revenue - sum(sold_items[:int(quantity)])
            agent.balance += total_revenue
            return {
                "status": "success",
                "action": "sell_when_feasible_2",
                "details": {
                    "price_per_unit": stock_prices[t],
                    "quantity": quantity,
                    "total_revenue": total_revenue,
                    "profit": profit,
                    "balance": agent.balance,
                    "inventory": agent.inventory
                }
            }
        return {
            "status": "skipped",
            "action": "sell_when_feasible_2",
            "details": {"reason": "No inventory or divergence signal"}
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
