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
            self.agent.actor.model.save_weights(f'saved_models/DDPG_ep{episode}_actor.h5')
            self.agent.critic.model.save_weights(f'saved_models/DDPG_ep{episode}_critic.h5')
        elif model_name == 'DDQN':
            self.agent.model.save(f'saved_models/DDQN_ep{episode}.h5')

    def execute_action(self, action, actions, stock_prices, t):
        if action == 0:  # hold
            return ActionService.hold(actions, self.agent, stock_prices, t)
        elif action == 1:  # buy
            return ActionService.buy(t, self.agent, stock_prices)
        elif action == 2:  # sell
            return ActionService.sell(t, self.agent, stock_prices)

    def calculate_reward(self, stock_prices, t, previous_portfolio_value):
        return RewardService.calculate_reward(self.agent, stock_prices, t, previous_portfolio_value)