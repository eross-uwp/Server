"""
__Author__: Nate Braukhoff
"""
from knowledge_base import KnowledgeBase

import pandas as pd
import itertools as it
import copy


# Todo: Need to implement how to get the probability when their is not enough data.


class CPTBuilder:
    def __init__(self, data, scale):
        columns = list(data.columns.values)

        self._owner = columns[0]
        self._parents = columns
        self._owner_data = data
        self._scale = scale

        columns.remove(self._owner)

    def build_with_no_parents(self):
        table = self._make_columns()
        row = []
        for i in range(0, len(self._scale)):
            # get probability
            print()
        table.loc[0] = row

    def _make_columns(self):
        """
        This method will create a DataFrame with the parents name followed by the scale.
        :return: DataFrame
        """
        columns = copy.deepcopy(self._parents)
        columns.extend(self._scale)

        return pd.DataFrame(columns=columns)

    def calculate_probability(self, combination, prediction):
        print()
        # apply filter with combination
        # count predictions in owner column

    def _apply_combination_filter(self, combination):
        """
        This method will filter data that only match the combination of data for the owner's parents. After it will
        return the data of the owner's column.
        :param combination: list
        :return: DataFrame
        """
        df = self._owner_data
        for i in range(0, len(self._parents)):
            temp_df = df[df[self._parents[i]] == combination[i]]
            if temp_df.shape[0] == 0:  # change to shape[0] >= 20
                return temp_df
            df = temp_df
        return df[self._owner]
