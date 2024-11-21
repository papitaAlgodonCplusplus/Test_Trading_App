class RewardService:
    @staticmethod
    def calculate_reward(agent, current_portfolio_value, initial_balance):
        # Reward is based on monthly profit
        return current_portfolio_value - initial_balance
