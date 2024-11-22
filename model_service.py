# model_service.py
import importlib
from tensorflow.keras.models import load_model

class ModelService:
    @staticmethod
    def load_model(model_name, state_dim=None, balance=None, best_model_path=None):
        """
        Loads the model dynamically based on the model name. 
        If a best_model_path is provided, load the pre-trained model.

        Parameters:
            model_name (str): The name of the model to load.
            state_dim (int): The dimensionality of the input state (optional, for creating new models).
            balance (float): Initial balance (optional, for creating new models).
            best_model_path (str): Path to the saved best model (if exists).

        Returns:
            An instance of the Agent class.
        """
        if best_model_path:
            print(f"Loading best model from: {best_model_path}")
            model_instance = importlib.import_module(f'agents.{model_name}')
            return model_instance.Agent(state_dim=state_dim, balance=balance, model_name=model_name, loaded=True)

        print(f"Loading new model: {model_name}")
        model_instance = importlib.import_module(f'agents.{model_name}')
        return model_instance.Agent(state_dim=state_dim, balance=balance)
