
import time
from agent_service import AgentService
from state_service import StateService
from reward_service import RewardService
from model_service import ModelService
import numpy as np   
from queue_manager import data_queue
from utils import treasury_bond_monthly_return_rate
from utils import update_plot

def train_model(model_name, _, window_size, num_episode, initial_balance, stock_prices, dates, key_levels):
    trading_period = len(stock_prices) - 1
    returns_across_episodes = []
    num_experience_replay = 0
    action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell', 3: 'Place Pending Buy', 4: 'Place Pending Sell'}

    AgentClass = ModelService.load_model(model_name)
    agent = AgentClass(state_dim=window_size + 3, balance=initial_balance)
    agent_service = AgentService(agent, balance=initial_balance)

    best_portfolio_return = float('-inf')
    start_time = time.time()
    for e in range(1, num_episode + 1):
        print(f'\nEpisode: {e}/{num_episode}')
        monthly_start_index = 0
        agent_service.reset()
        x_vals, y_vals = [], []
        hold_signals, buy_signals, sell_signals, place_buy_signals, place_sell_signals = [], [], [], [], []
        dates_list = list(dates)
        state = StateService.generate_state(
            0, window_size, stock_prices, agent_service.balance, len(agent_service.inventory))
        agent_service.monthly_portfolio_value = agent_service.balance

        for t in range(1, trading_period + 1):
            previous_inventory_value = sum(stock_prices[monthly_start_index] * quantity for quantity in agent_service.inventory)
            if dates[t].month != dates[monthly_start_index].month or t == trading_period:
                realized_gain = agent_service.balance - agent_service.balance
                unrealized_gain = sum(stock_prices[t - 1] * quantity for quantity in agent_service.inventory) - previous_inventory_value
                reward = realized_gain + unrealized_gain

                agent_service.monthly_portfolio_value = agent_service.current_portfolio_value
                agent_service.current_portfolio_value = agent_service.monthly_portfolio_value
                agent_service.balance += sum(stock_prices[t - 1] * quantity for quantity in agent_service.inventory)
                agent_service.inventory = []
                monthly_start_index = t
                agent_service.total_gains += agent_service.monthly_gains
                agent_service.total_losses += agent_service.monthly_losses
                agent_service.monthly_gains = 0
                agent_service.monthly_losses = 0
            else:
                reward = 0

            next_state = StateService.generate_state(
                t, window_size, stock_prices, agent_service.balance, len(agent_service.inventory))

            if model_name == 'DDPG':
                actions = agent_service.act(state, t)
                action = np.argmax(actions)
            else:
                actions = agent.model.predict(state)[0]
                action = agent_service.act(state)

            execute_simulation(agent, agent_service, actions, stock_prices, dates, trading_period, x_vals, y_vals, 
                               hold_signals, buy_signals, sell_signals, action, dates_list, e, returns_across_episodes, 
                               num_experience_replay, t, action_dict, next_state, state, reward, t == trading_period, 
                               key_levels, place_buy_signals, place_sell_signals)

        if returns_across_episodes[-1] > best_portfolio_return:
            best_portfolio_return = returns_across_episodes[-1]
            agent_service.save_model(model_name, e)
            print('Best model saved')

    print('total training time: {0:.2f} min'.format(
        (time.time() - start_time) / 60))
    return returns_across_episodes


def execute_simulation(agent, agent_service, actions, stock_prices, dates, trading_period, x_vals, y_vals, 
                       hold_signals, buy_signals, sell_signals, action, dates_list, e, returns_across_episodes, 
                       num_experience_replay, t, action_dict, next_state, state, reward, done, key_levels, place_buy_signals, place_sell_signals):
    act_and_calculate(agent, agent_service, actions, stock_prices, t, action, key_levels)

    updated_data = update_plot(t, stock_prices[t], action, x_vals, y_vals, hold_signals, buy_signals, sell_signals, 
                               agent_service.current_portfolio_value, agent_service.monthly_portfolio_value, dates, 
                               agent_service.total_gains, agent_service.total_losses, agent_service.monthly_gains, 
                               agent_service.monthly_losses, agent_service.initial_portfolio_value, place_buy_signals, place_sell_signals)

    data_queue.put(updated_data)

    state = next_state
    done = t == trading_period
    agent_service.remember(state, actions, reward, next_state, done)

    if done:
        portfolio_return = RewardService.calculate_reward(agent, agent_service.current_portfolio_value, 
                                                          agent_service.monthly_portfolio_value)
        returns_across_episodes.append(portfolio_return)


def act_and_calculate(agent, agent_service, actions, stock_prices, t, action, key_levels):
    print("Executing action: ", action, " in ", actions, " at time ", t)
    execution_result = agent_service.execute_action(action, stock_prices, t, key_levels)
    temp_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance

    if temp_portfolio_value > agent_service.current_portfolio_value:
        agent_service.monthly_gains += temp_portfolio_value - agent_service.current_portfolio_value
    elif temp_portfolio_value < agent_service.current_portfolio_value:
        agent_service.monthly_losses += agent_service.current_portfolio_value - temp_portfolio_value
    agent_service.current_portfolio_value = temp_portfolio_value

    if execution_result is None:
        reward = -treasury_bond_monthly_return_rate() * agent_service.balance
    else:
        if isinstance(execution_result, tuple):
            actions = execution_result[1]
            execution_result = execution_result[0]
