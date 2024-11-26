import os
from agent_service import AgentService
from state_service import StateService
from reward_service import RewardService
from model_service import ModelService
import numpy as np
from dash_app import reset_everything
from queue_manager import data_queue
from utils import update_plot
from market_intelligence import obtain_forbidden_actions

global best_portfolio_return

def train_model(using_dash_app, model_name, _, window_size, num_episode, initial_balance, stock_prices, dates, key_levels):
    """
    Train the trading model to optimize profits, with stop-loss and take-profit based on user-defined risk.

    Args:
        model_name: Name of the model being trained.
        window_size: Window size for the state representation.
        num_episode: Number of training episodes.
        initial_balance: Starting balance for the agent.
        stock_prices: Historical stock prices for training.
        dates: Corresponding dates for the stock prices.
        key_levels: Key levels for decision-making.

    Returns:
        List of returns across episodes.
    """
    global best_portfolio_return
    best_portfolio_return = float('-inf')
    trading_period = len(stock_prices) - 1
    returns_across_episodes = []

    # Ask the user for max risk percentage
    # max_risk = float(input("Enter your maximum balance risk percentage (e.g., 10 for 10%): "))
    max_risk = 100

    # Initialize variables
    initial_balance_value = initial_balance * stock_prices[0]
    stop_loss = initial_balance_value * (1 - max_risk / 100)
    take_profit = initial_balance_value * 1.05
    print(f"Stop-loss set to {stop_loss:.2f}. Take-profit set to {take_profit:.2f}.")
    agent = load_model_and_reset_plot(False, model_name, window_size, initial_balance_value)
    agent_service = AgentService(agent, balance=initial_balance_value)

    for e in range(1, num_episode + 1):
        print(f'\nEpisode: {e}/{num_episode}')
        # Initialize state and signals
        profit_vault = 0  # Initialize profit vault
        agent_service.reset()
        x_vals, y_vals = [], []
        hold_signals, buy_signals, sell_signals, place_buy_signals, place_sell_signals, pending_orders_signals = [], [], [], [], [], []
        reward = 0
        state = StateService.generate_state(
            0, window_size, stock_prices, agent_service.agent.balance, len(agent_service.agent.inventory)
        )

        # Trading loop
        for t in range(1, trading_period + 1):
            next_state = StateService.generate_state(
                t, window_size, stock_prices, agent_service.agent.balance, len(agent_service.agent.inventory)
            )

            actions = agent.model.predict(state, verbose=0)[0]
            risk = initial_balance * (max_risk / 100) - len(agent_service.agent.inventory)

            action_dict = {
                0: 'Hold',
                1: 'Buy Instantly',
                2: 'Sell Instantly',
            }

            forbidden_actions = []
            if len(agent_service.agent.inventory) == 0:
                forbidden_actions.append(2)  # Prevent selling if no inventory
            if risk <= 0:
                forbidden_actions.append(1) # Prevent buying if risk is too high
              
            action, quantity = agent_service.act(state, risk, forbidden_actions)

            # Execute the action and update the simulation
            execution_result = execute_simulation(
                using_dash_app, agent, agent_service, actions, stock_prices, dates, trading_period, x_vals, y_vals,
                hold_signals, buy_signals, sell_signals, action, e, returns_across_episodes, 0, t, action_dict, next_state,
                state, reward, t == trading_period, key_levels, place_buy_signals, place_sell_signals, pending_orders_signals,
                model_name, quantity
            )

            # Update profit vault if balance exceeds the initial balance
            if agent_service.agent.balance > initial_balance_value:
                profit = agent_service.agent.balance - initial_balance_value
                profit_vault += profit
                initial_balance_value += profit  # Adjust initial balance for future comparisons

            # Calculate scaled reward
            reward = calculate_reward(agent_service, initial_balance_value, 0, profit_vault)

            from colorama import Fore
            print(
                Fore.WHITE + "Date: " + Fore.RESET +
                Fore.CYAN + f"{dates[t]}" + Fore.RESET +
                Fore.WHITE + " - Reward: " + Fore.RESET +
                Fore.GREEN + f"{reward:.2f}" + Fore.RESET +
                Fore.WHITE + " - Balance: " + Fore.RESET +
                Fore.MAGENTA + f"{agent_service.agent.balance:.2f}" + Fore.RESET +
                Fore.WHITE + " - Inventory: " + Fore.RESET +
                Fore.YELLOW + f"{len(agent_service.agent.inventory)}" + Fore.RESET +
                Fore.WHITE + " - Action: " + Fore.RESET +
                Fore.RED + f"{action_dict[action]}" + Fore.RESET +
                Fore.WHITE + " - Profit Vault: " + Fore.RESET +
                Fore.CYAN + f"{profit_vault:.2f}" + Fore.RESET
            )

            # Check stop-loss and take-profit conditions
            if agent_service.agent.balance <= stop_loss:
                print(Fore.RED + "Training stopped early due to stop-loss being hit!" + Fore.RESET)
                return returns_across_episodes
            
            if agent_service.agent.balance + profit_vault >= (take_profit):
                print(Fore.GREEN + "Training stopped early due to take-profit being achieved!" + Fore.RESET)
                
                portfolio_return = (agent_service.agent.balance - initial_balance_value) + profit_vault
                if portfolio_return > best_portfolio_return:
                    best_portfolio_return = portfolio_return
                    agent_service.agent.save_models()
                    print('Best model saved with portfolio return:', best_portfolio_return)
                
                return returns_across_episodes

        # Save the best model if portfolio return improves
        agent_service.execute_action(stock_prices, t, key_levels, 0, 2, False)
        print(
              Fore.WHITE + "Date: " + Fore.RESET +
              Fore.CYAN + f"{dates[t]}" + Fore.RESET +
              Fore.WHITE + " - Reward: " + Fore.RESET +
              Fore.GREEN + f"{reward:.2f}" + Fore.RESET +
              Fore.WHITE + " - Balance: " + Fore.RESET +
              Fore.MAGENTA + f"{agent_service.agent.balance:.2f}" + Fore.RESET +
              Fore.WHITE + " - Inventory: " + Fore.RESET +
              Fore.YELLOW + f"{len(agent_service.agent.inventory)}" + Fore.RESET +
              Fore.WHITE + " - Action: " + Fore.RESET +
              Fore.RED + f"{action_dict[action]}" + Fore.RESET +
              Fore.WHITE + " - Profit Vault: " + Fore.RESET +
              Fore.CYAN + f"{profit_vault:.2f}" + Fore.RESET
          )
        portfolio_return = (agent_service.agent.balance - initial_balance_value) + profit_vault
        
        if portfolio_return > best_portfolio_return:
            best_portfolio_return = portfolio_return
            agent_service.agent.save_models()
            print('Best model saved with portfolio return:', best_portfolio_return)

        returns_across_episodes.append(portfolio_return)

    return returns_across_episodes

def execute_simulation(using_dash_app, agent, agent_service, actions, stock_prices, dates, trading_period, x_vals, y_vals, hold_signals, buy_signals, sell_signals, action, e, returns_across_episodes,  ___, t, __, next_state, state, reward, done, key_levels, place_buy_signals, place_sell_signals, pending_orders_signals, _, quantity):
    execution_result = act_and_calculate(agent, agent_service, actions,
                          stock_prices, t, action, key_levels, quantity)

    if (using_dash_app):
        updated_data = update_plot(
            t, stock_prices[t], action, x_vals, y_vals, hold_signals, buy_signals, sell_signals,
            agent_service.agent.balance, agent_service.initial_portfolio_value, dates, agent_service.total_gains, agent_service.total_losses,
            place_buy_signals, place_sell_signals, pending_orders_signals, sell_signals,
            [], [], reward
        )

        data_queue.put(updated_data)

    state = next_state
    done = t == trading_period
    agent_service.remember(state, actions, reward, next_state, done)
    agent_service.experience_replay()

    if done:
        returns_across_episodes.append(reward)

    return execution_result

def act_and_calculate(__, agent_service, _, stock_prices, t, action, kl, quantity):
    execution_result = agent_service.execute_action(stock_prices, t, kl, quantity, action, False)
    return execution_result

def standardize_value(value, initial_balance_value):
    min_value = initial_balance_value * 0  # This is effectively 0
    max_value = initial_balance_value * 1.5
    standardized_value = (value - min_value) / (max_value - min_value) - 1
    return standardized_value + 1

def calculate_reward(agent_service, initial_balance_value, _, profit_vault):
    """
    Calculate reward based on current balance, vault balance, and initial balance,
    scaled proportionally to [-1, 1].
    """
    # Combine agent balance and vault balance
    total_balance = agent_service.agent.balance + profit_vault

    # Calculate deviation from the initial balance
    deviation = total_balance - initial_balance_value

    # Scale the deviation to the range [-1, 1]
    return standardize_value(deviation, initial_balance_value)

def load_model_and_reset_plot(avaliable, model_name, window_size, initial_balance_value):
    """
    Load the best model if available, or initialize a new one, and reset the plot.

    Args:
        avaliable: Whether the best model is available.
        model_name: Name of the model.
        window_size: Window size for state representation.
        initial_balance_value: Initial balance for the agent.

    Returns:
        The loaded or newly initialized agent model.
    """
    reset_everything()
    model = ModelService.load_model(
            model_name=model_name,
            state_dim=window_size + 3,
            balance=initial_balance_value)
    
    if avaliable:
        model.load_models()
    
    return model