import time
import os
from agent_service import AgentService
from state_service import StateService
from reward_service import RewardService
from model_service import ModelService
import numpy as np   
from queue_manager import data_queue
from utils import treasury_bond_monthly_return_rate
from utils import update_plot
global best_portfolio_return

def train_model(model_name, _, window_size, num_episode, initial_balance, stock_prices, dates, key_levels):
    best_portfolio_return = float('-inf')
    trading_period = len(stock_prices) - 1
    returns_across_episodes = []
    num_experience_replay = 0
    action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell', 3: 'Place Pending Buy', 4: 'Place Pending Sell'}
    best_model_path = f"saved_models/best_{model_name}.h5"

    start_time = time.time()
    for e in range(1, num_episode + 1):
        reset_data_point = {
            "signal": "RESET",
        }
        data_queue.put(reset_data_point)
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            agent = ModelService.load_model(
                model_name='DDQN', 
                state_dim=window_size + 3, 
                balance=initial_balance, 
                best_model_path=best_model_path
            )
        else:
            print(f"No best model found. Initializing new model: {model_name}")
            agent = ModelService.load_model(
                model_name=model_name, 
                state_dim=window_size + 3, 
                balance=initial_balance
            )
        agent_service = AgentService(agent, balance=initial_balance)
        print(f'\nEpisode: {e}/{num_episode}')
        monthly_start_index = 0
        agent_service.reset()
        x_vals, y_vals = [], []
        hold_signals, buy_signals, sell_signals, place_buy_signals, place_sell_signals, pending_orders_signals = [], [], [], [], [], []
        dates_list = list(dates)
        state = StateService.generate_state(
            0, window_size, stock_prices, agent.balance, len(agent.inventory))
        agent_service.monthly_portfolio_value = agent.balance

        for t in range(1, trading_period + 1):
            previous_inventory_value = sum(stock_prices[monthly_start_index] * quantity for quantity in agent.inventory)

            # Handle end-of-month or final step logic
            if dates[t].month != dates[monthly_start_index].month or t == trading_period:
                # Calculate realized gain (cash changes + pending buy commitments)
                realized_gain = (
                    agent_service.current_portfolio_value
                    - agent_service.monthly_portfolio_value
                    - sum(order[1] for order in agent_service.pending_orders if order[0] == "buy")
                )

                # Calculate unrealized gain (inventory value + pending sells)
                unrealized_gain = (
                    sum(stock_prices[t - 1] * quantity for quantity in agent.inventory)
                    + sum(order[1] for order in agent_service.pending_orders if order[0] == "sell")
                    - previous_inventory_value
                )

                # Total reward
                reward = realized_gain + unrealized_gain

                # Update monthly stats
                agent_service.monthly_portfolio_value = agent_service.current_portfolio_value
                agent.balance += sum(stock_prices[t - 1] * quantity for quantity in agent.inventory)
                agent.inventory = []
                monthly_start_index = t

                # Update cumulative gains/losses
                agent_service.total_gains += agent_service.monthly_gains
                agent_service.total_losses += agent_service.monthly_losses
                agent_service.monthly_gains = 0
                agent_service.monthly_losses = 0

            # Provide smaller reward signals in non-month-end steps
            else:
                # Estimate incremental reward based on short-term portfolio changes
                incremental_realized_gain = (
                    agent_service.current_portfolio_value - agent_service.monthly_portfolio_value
                )
                incremental_unrealized_gain = (
                    sum(stock_prices[t] * quantity for quantity in agent.inventory) - previous_inventory_value
                )
                reward = incremental_realized_gain + incremental_unrealized_gain

            next_state = StateService.generate_state(
                t, window_size, stock_prices, agent.balance, len(agent.inventory))

            if model_name == 'DDPG':
                actions = agent_service.act(state, t)
                action = np.argmax(actions)
            else:
                actions = agent.model.predict(state, verbose=0)[0]
                action = agent_service.act(state)

            execute_simulation(agent, agent_service, actions, stock_prices, dates, trading_period, x_vals, y_vals, 
                               hold_signals, buy_signals, sell_signals, action, dates_list, e, returns_across_episodes, 
                               num_experience_replay, t, action_dict, next_state, state, reward, t == trading_period, 
                               key_levels, place_buy_signals, place_sell_signals, pending_orders_signals, model_name)

        if returns_across_episodes[-1] > best_portfolio_return:
            best_portfolio_return = returns_across_episodes[-1]
            agent_service.save_model(model_name, e)
            print('Best model saved')

    print('total training time: {0:.2f} min'.format(
        (time.time() - start_time) / 60))
    return returns_across_episodes


def execute_simulation(agent, agent_service, actions, stock_prices, dates, trading_period, x_vals, y_vals, 
                       hold_signals, buy_signals, sell_signals, action, dates_list, e, returns_across_episodes, 
                       num_experience_replay, t, action_dict, next_state, state, reward, done, key_levels, place_buy_signals, place_sell_signals, pending_orders_signals, model_name):
    
    execution_result = act_and_calculate(agent, agent_service, actions, stock_prices, t, action, key_levels)

    updated_data = update_plot(t, stock_prices[t], action, x_vals, y_vals, hold_signals, buy_signals, sell_signals, 
                               agent_service.current_portfolio_value, agent_service.monthly_portfolio_value, dates, 
                               agent_service.total_gains, agent_service.total_losses, agent_service.monthly_gains, 
                               agent_service.monthly_losses, agent_service.initial_portfolio_value, place_buy_signals, place_sell_signals, pending_orders_signals, execution_result)

    
    data_queue.put(updated_data)

    state = next_state
    done = t == trading_period
    agent_service.remember(state, actions, reward, next_state, done)

    if done:
        portfolio_return = RewardService.calculate_reward(agent, agent_service.current_portfolio_value, 
                                                          agent_service.monthly_portfolio_value)
        reward = 0
        returns_across_episodes.append(portfolio_return)

    else: 
        # Give gradual reward
        reward = agent_service.current_portfolio_value - agent_service.monthly_portfolio_value

def act_and_calculate(agent, agent_service, _, stock_prices, t, action, key_levels):
    # Execute the selected action
    execution_result = agent_service.execute_action(action, stock_prices, t, key_levels)

    # Calculate reserved funds for pending buy orders
    reserved_funds = sum(order[1] for order in agent_service.pending_orders if order[0] == "buy")

    # Compute temporary portfolio value (excluding reserved funds)
    temp_portfolio_value = (
        agent.balance - reserved_funds
        + sum(stock_prices[t] * quantity for quantity in agent.inventory)
    )

    # Update gains/losses
    if temp_portfolio_value > agent_service.current_portfolio_value:
        agent_service.monthly_gains += temp_portfolio_value - agent_service.current_portfolio_value
    elif temp_portfolio_value < agent_service.current_portfolio_value:
        agent_service.monthly_losses += agent_service.current_portfolio_value - temp_portfolio_value

    # Update current portfolio value
    agent_service.current_portfolio_value = temp_portfolio_value

    return execution_result
