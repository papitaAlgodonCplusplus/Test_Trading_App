import pandas as pd

class DataStorage:
    def __init__(self):
        self.dataframe = None
        self.last_action = ""
        self.last_closing_date = None

    def add_dataframe(self, dataframe):
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("The provided data is not a pandas DataFrame")
        self.dataframe = dataframe

    def get_dataframe(self):
        return self.dataframe if self.dataframe is not None else None

    def remove_dataframe(self):
        self.dataframe = None

    def has_dataframe(self):
        return self.dataframe is not None

    def set_last_action(self, action):
        self.last_action = action

    def get_last_action(self):
        return self.last_action
    
    def set_last_closing_date(self, date):
        self.last_closing_date = date
    
    def get_last_closing_date(self):
        return self.last_closing_date
    
