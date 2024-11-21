# agent_service.py
import numpy as np
from action_service import ActionService
from reward_service import RewardService


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

    def reset(self):
        self.agent.reset()

    def act(self, state, t=None):
        if t is not None:
            return self.agent.act(state, t)
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
            self.agent.model.save(f'saved_models/DDQN_ep{episode}.h5')

    def execute_action(self, action, stock_prices, t, key_levels=None):
        """
        Executes the selected action based on the action space.
        
        Parameters:
            action (int): Index of the action to be executed.
            stock_prices (list): List of stock prices.
            t (int): Current time step.
            key_levels (list): Key liquidity levels for pending orders (if applicable).

        Returns:
            str: A message describing the result of the executed action.
        """
        if action == 0:  # Hold
            return ActionService.hold(self.agent, stock_prices, t)
        elif action == 1:  # Buy
            return ActionService.buy(self.agent, stock_prices, t)
        elif action == 2:  # Sell
            return ActionService.sell(self.agent, stock_prices, t)
        elif action == 3:  # Place Pending Buy
            if key_levels is None:
                return "Pending Buy Skipped (Missing Key Levels)"
            return ActionService.place_pending_buy(self.agent, stock_prices, t, key_levels)
        elif action == 4:  # Place Pending Sell
            if key_levels is None:
                return "Pending Sell Skipped (Missing Key Levels)"
            return ActionService.place_pending_sell(self.agent, stock_prices, t, key_levels)
        else:
            return f"Invalid Action: {action}"

    def calculate_reward(self, stock_prices, t, previous_portfolio_value):
        return RewardService.calculate_reward(self.agent, stock_prices, t, previous_portfolio_value)
