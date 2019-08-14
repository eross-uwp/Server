from node import Node
from conditional_probability_table import ConditionalProbabilityTable

import pandas as pd
import itertools as it


class CPTBuilder:
    def __init__(self, name, parent_names, kb):
        if parent_names is None:
            parent_names = []
        self._table = pd.DataFrame()
        self._node_name = name
        self._parents = parent_names
        self._knowledge_base = kb

    def build(self):
        self._build_columns()
        if len(self._parents) == 0:
            row = []
            for i in range(0, len(self._knowledge_base.get_scale())):
                row.append(self.get_probability_of_combination(self._knowledge_base.get_data(),
                                                               self._knowledge_base.get_scale()[i]))
            self._table.loc[0] = row
        else:
            combination = self.get_all_combination(self._knowledge_base.get_scale())
            for i in range(0, len(combination)):
                row = []
                row.extend(combination[i])
                for value in self._knowledge_base.get_scale():
                    row.append(self.get_probability_of_combination(self.filter_data((combination[i]), value),
                                                                   self._knowledge_base.get_scale()[i]))
                self._table.loc[i] = row

        return self._table

    def _build_columns(self):
        """
        This method will add all the node's parent's names as columns in the DataFrame. After the parent's
        each value in the scale will also be a column after the parents name.
        """
        scale = self._knowledge_base.get_scale()
        columns = self._parents
        columns.extend(scale)

        self._table = pd.DataFrame(columns=columns)

    def filter_data(self, data, combination):
        """
        filter_data will filter the data frame based on the combination of the parents states.
        :param data: DataFrame
        :param combination: list of strings
        :return: Data Frame
        """
        df = data
        for i in range(0, len(self._parents)):
            temp_df = df[df[self._parents[i]] == combination[i]]
            if temp_df.shape[0] != 0: # change to shape[0] >= 20
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
        :param scale: list of items
        :return: list of combinations
        """
        combination_list = []
        for i in list(it.product(scale, repeat=len(self._parents))):
            combination_list.append(i)

        return combination_list
