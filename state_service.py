# state_service.py
from utils import generate_combined_state

class StateService:
    @staticmethod
    def generate_state(t, window_size, stock_prices, balance, inventory_length):
        return generate_combined_state(t, window_size, stock_prices, balance, inventory_length)