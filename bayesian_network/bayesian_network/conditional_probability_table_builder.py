"""
__Author__: Nate Braukhoff
"""
from knowledge_base import KnowledgeBase

import pandas as pd
import itertools as it
import copy

# Todo: Need to implement how to get the probability when their is not enough data.


class CPTBuilder:
    def __init__(self, name, parent_names):
        if parent_names is None:
            parent_names = []
        self._table = pd.DataFrame()
        self._node_name = name
        self._parents = parent_names

    def get_table(self):
        return self._table

    def build(self, data, scale):
        """
        This method will build the conditional probability table
        :return: DataFrame
        """
        self.build_columns(scale)
        if len(self._parents) == 0:
            self._build_with_no_parents(scale, data)
        else:
            self._build_with_parents(scale, data)

        return self._table

    def build_columns(self, scale):
        """
        This method will add all the node's parent's names as columns in the DataFrame. After the parent's
        each value in the scale will also be a column after the parents name.
        """
        columns = copy.deepcopy(self._parents)
        columns.extend(scale)

        self._table = pd.DataFrame(columns=columns)

    def filter_data(self, combination, data):
        """
        filter_data will filter the data frame based on the combination of the parents states.
        :param data:
        :param combination: list of strings
        :return: Data Frame
        """
        df = data
        for i in range(0, len(self._parents)):
            temp_df = df[df[self._parents[i]] == combination[i]]
            if temp_df.shape[0] == 0: # change to shape[0] >= 20
                return temp_df
            df = temp_df
        return df

    def get_probability_of_combination(self, data, predict):
        """
        This method will return the probability of the node's state given it's parents' states.
        :param data: DataFrame
        :param predict: state of Node
        :return: float
        """
        if data.shape[0] == 0:
            return 0
        occurrence = len(data[data[self._node_name] == predict])
        return occurrence / data.shape[0]

    def get_all_combination(self, scale):
        """
        get_all_combination takes in a list of items, then returns all possible combinations of each item.
        :return: list of combinations
        """
        # todo: add scale are a variable
        combination_list = []
        for i in list(it.product(scale, repeat=len(self._parents))):
            combination_list.append(i)

        return combination_list

    def _build_with_no_parents(self, scale, data):
        """
        This method will build a table when the node has no parents.
        """
        row = []
        for i in range(0, len(scale)):
            row.append(self.get_probability_of_combination(data, scale[i]))
        self._table.loc[0] = row

    def _build_with_parents(self, scale, data):
        """
        This method will build the table when a node has parents.
        """
        combination = self.get_all_combination(scale)
        for i in range(0, len(combination)):
            filter_data = self.filter_data(combination[i], data)
            if filter_data.shape[0] > 0:
                row = []
                row.extend(combination[i])
                for value in scale:
                    row.append(self.get_probability_of_combination(filter_data, value))
                self._table.loc[i] = row
