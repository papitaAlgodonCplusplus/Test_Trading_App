# agent_service.py
import numpy as np
from action_service import ActionService
from reward_service import RewardService
from market_analyzer import (
    get_trend, get_order_blocks, get_liquidity_zones, get_divergence_signal
)

class AgentService:
    def __init__(self, agent, balance):
        self.agent = agent
        self.initial_portfolio_value = balance
        self.balance = balance
        self.inventory = []
        self.current_portfolio_value = balance
        self.monthly_portfolio_value = balance
        self.current_portfolio_values = []
        self.monthly_portfolio_values = []
        self.portfolio_values = []
        self.return_rates = []
        self.total_gains = 0
        self.total_losses = 0
        self.monthly_gains = 0
        self.monthly_losses = 0
        self.pending_orders = []

    def reset(self):
        self.agent.reset()

    def act(self, state, max_quantity=None, forbidden_actions=[]):
        if max_quantity is not None:
            return self.agent.act(state, max_quantity, forbidden_actions)
        return self.agent.act(state)

    def remember(self, state, actions, reward, next_state, done):
        self.agent.remember(state, actions, reward, next_state, done)

    def experience_replay(self):
        return self.agent.experience_replay()

    def save_model(self, model_name, episode):
        if model_name == 'DQN':
            self.agent.model.save(f'saved_models/DQN_ep{episode}.h5')
        elif model_name == 'DDPG':
            self.agent.actor.model.save_weights(
                f'saved_models/DDPG_ep{episode}_actor.h5')
            self.agent.critic.model.save_weights(
                f'saved_models/DDPG_ep{episode}_critic.h5')
        elif model_name == 'DDQN':
            self.agent.model.save(f'saved_models/best_{model_name}.h5')
            self.agent.model_target.save(f'saved_models/best_{model_name}_target.h5')

    def execute_action(self, stock_prices, t, _, quantity, action, optimize=True):
        """
        Executes an action based on market analysis with a confidence threshold if optimize is enabled.

        Args:
            stock_prices: Historical stock prices.
            t: Current time step.
            key_levels: Key levels for market analysis.
            quantity: Quantity to buy or sell.
            action: Default action if no confident action is found.
            optimize: If True, use the weighted system; otherwise, execute the default action.

        Returns:
            dict: Execution result with action details.
        """
        if not optimize:
            # Execute the default action without optimization
            if action == 0:  # Hold
                return ActionService.hold(self.agent, stock_prices, t)
            elif action == 1:  # Buy Instantly
                return ActionService.buy(self.agent, stock_prices, t, quantity)
            elif action == 2:  # Sell Instantly
                return ActionService.sell_instant(self.agent, stock_prices, t, quantity)
           
            # Fallback for unexpected cases
            return {"status": "error", "action": "none", "details": {"reason": "Invalid Action"}}

        # Analyze market conditions
        trend = get_trend(stock_prices, t)
        order_blocks = get_order_blocks(stock_prices, t)
        liquidity_zones = get_liquidity_zones(stock_prices, t)
        divergence_signal = get_divergence_signal(stock_prices, t)

        # Initialize weights
        action_weights = {
            0: {"action": "hold", "weight": 0.5},  # Default neutral weight
            1: {"action": "buy_instant", "weight": 0},
            2: {"action": "buy_when_demand", "weight": 0},
            3: {"action": "sell_instant", "weight": 0},
            4: {"action": "sell_when_feasible_1", "weight": 0},
            5: {"action": "sell_when_feasible_2", "weight": 0},
        }

        # Assign weights based on trend analysis
        if trend == "bullish":
            action_weights[1]["weight"] += 0.6  # Increase buy weight
            action_weights[2]["weight"] += 0.5  # Demand zone buy
        elif trend == "bearish":
            action_weights[3]["weight"] += 0.6  # Increase sell weight
            action_weights[4]["weight"] += 0.5  # Liquidity zone sell

        # Incorporate order blocks
        if order_blocks.get("demand_zone"):
            action_weights[2]["weight"] += 0.3  # Favor buy in demand zone
        if order_blocks.get("supply_zone"):
            action_weights[4]["weight"] += 0.3  # Favor sell in supply zone

        # Incorporate liquidity zones
        if liquidity_zones.get("bullish_liquidity"):
            action_weights[1]["weight"] += 0.4  # Favor buy for bullish liquidity
        if liquidity_zones.get("bearish_liquidity"):
            action_weights[3]["weight"] += 0.4  # Favor sell for bearish liquidity

        # Incorporate divergence signals
        if divergence_signal:
            action_weights[5]["weight"] += 0.7  # Favor divergence-based sell

        # Normalize weights to sum to 1
        total_weight = sum(a["weight"] for a in action_weights.values())
        for a in action_weights.values():
            a["weight"] /= total_weight

        # Choose action with the highest weight above confidence threshold
        chosen_action = max(
            action_weights.items(),
            key=lambda x: x[1]["weight"] if x[1]["weight"] >= 0.8 else -1,
        )[0]

        # Default to the provided action if no high-confidence action is found
        if action_weights[chosen_action]["weight"] < 0.8:
            chosen_action = action

        # Execute the chosen action
        if chosen_action == 0:  # Hold
            return ActionService.hold(self.agent, stock_prices, t)
        elif chosen_action == 1:  # Buy Instantly
            return ActionService.buy(self.agent, stock_prices, t, quantity)
        elif chosen_action == 2:  # Buy When Demand
            return ActionService.buy_when_demand(self.agent, stock_prices, t, quantity)
        elif chosen_action == 3:  # Sell Instantly
            return ActionService.sell_instant(self.agent, stock_prices, t, quantity)
        elif chosen_action == 4:  # Sell When Feasible 1
            return ActionService.sell_when_feasible_1(self.agent, stock_prices, t, quantity)
        elif chosen_action == 5:  # Sell When Feasible 2
            return ActionService.sell_when_feasible_2(self.agent, stock_prices, t, quantity)

        # Fallback for unexpected cases
        return {"status": "error", "action": "none", "details": {"reason": "Invalid Action"}}
