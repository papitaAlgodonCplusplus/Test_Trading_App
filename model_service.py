# model_service.py
import importlib

class ModelService:
    @staticmethod
    def load_model(model_name):
        model = importlib.import_module(f'agents.{model_name}')
        return model.Agent