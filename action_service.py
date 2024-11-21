# action_service.py
import numpy as np

class ActionService:
    @staticmethod
    def hold(actions, agent, stock_prices, t):
        next_probable_action = np.argsort(actions)[1]
        if next_probable_action == 2 and len(agent.inventory) > 0:
            max_profit = stock_prices[t] - min(agent.inventory)
            if max_profit > 0:
                ActionService.sell(t, agent, stock_prices)
                actions[next_probable_action] = 1  # reset this action's value to the highest
                return 'Hold', actions

    @staticmethod
    def buy(t, agent, stock_prices):
        if agent.balance > stock_prices[t]:
            agent.balance -= stock_prices[t]
            agent.inventory.append(stock_prices[t])
            return 'Buy: ${:.2f}'.format(stock_prices[t])

    @staticmethod
    def sell(t, agent, stock_prices):
        if len(agent.inventory) > 0:
            agent.balance += stock_prices[t]
            bought_price = agent.inventory.pop(0)
            profit = stock_prices[t] - bought_price
            return 'Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit)