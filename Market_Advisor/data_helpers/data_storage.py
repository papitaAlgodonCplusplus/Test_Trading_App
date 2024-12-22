import pandas as pd

class DataStorage:
    def __init__(self):
        self.dataframe = None

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
