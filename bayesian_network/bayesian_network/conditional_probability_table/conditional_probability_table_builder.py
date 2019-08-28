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
        columns.remove(self._owner)

        self._parents = columns
        self._owner_data = data
        self._scale = scale

    def build_with_no_parents(self):
        """
        This method will build the Conditional Probability Table for a Node that has no parents
        :return: DataFrame
        """
        table = self._make_columns()
        table.loc[0] = self._add_probabilities_to_row(list(self._owner_data.values))
        return table

    def build_with_parents(self):
        table = self._make_columns()
        combinations = self._get_all_possible_combination()

        for i in range(0, len(combinations)):
            row = []
            combination = combinations[i]

            row.extend(combination)
            entry_list = list(self._apply_combination_filter(combination))
            row.extend(self._add_probabilities_to_row(entry_list))

            table.loc[i] = row
        return table

    def _make_columns(self):
        """
        This method will create a DataFrame with the parents name followed by the scale.
        :return: DataFrame
        """
        columns = copy.deepcopy(self._parents)
        columns.extend(self._scale)
        return pd.DataFrame(columns=columns)

    def _calculate_probability(self, entry_list, prediction):
        """
        This method calculate the probability of the owner's state based on the states of it's parents.
        :param entry_list: list
        :param prediction: char
        :return:
        """
        total_entries = len(entry_list)
        if total_entries == 0:  # change to shape[0] >= 20
            return 0
        occurrence = entry_list.count(prediction)
        return occurrence / total_entries

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
            if temp_df.shape[0] == 0:
                return temp_df
            df = temp_df
        return df[self._owner]

    def _get_all_possible_combination(self):
        """
        This method will return a list with all combination that is possible with the scale and the number of parents the
        owner has.
        :return: list of combinations
        """
        combination_list = []
        for i in list(it.product(self._scale, repeat=len(self._parents))):
            combination_list.append(i)

        return combination_list

    def _add_probabilities_to_row(self, entry_list):
        """
        This method will calculate the probability for each value in the scale
        :param entry_list:
        :return: list
        """
        row = []
        for i in range(0, len(self._scale)):
            probability = self._calculate_probability(entry_list, self._scale[i])
            row.append(probability)
        return row
