class RewardService:
    @staticmethod
    def calculate_reward(agent, current_portfolio_value, initial_balance):
        return current_portfolio_value - initial_balance
