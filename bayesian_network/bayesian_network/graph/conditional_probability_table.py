import pandas as pd


class ConditionalProbabilityTable:
    def __init__(self, data):
        self._data = data
        self._cpt = pd.DataFrame()

    # todo implement all methods from the Node class
