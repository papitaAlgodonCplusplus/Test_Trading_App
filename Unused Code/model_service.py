import importlib

class ModelService:
    @staticmethod
    def load_model(model_name, state_dim=None, balance=None, _=None):
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
        
        model_instance = importlib.import_module(f'agents.{model_name}')
        return model_instance.Agent(state_dim=state_dim, balance=balance)