import itertools

import pandas as pd

# Todo: Need to implement how to get the probability when their is not enough data.
"""
When there isn't enough data to make a accurate prediction, we will weigh each grades then normalized the probability.
See Adjusted_CPT_Tables.xlsx for more details. In the method filter_data there is a condition that needs to be changes. 
This right condition is next to the if statement and is commented out. 
"""


class ConditionalProbabilityTable:
    def __init__(self, node):
        """
        Constructs a Conditional Probability Table
        :param node: Node
        """
        self._node = node
        self._cpt = pd.DataFrame()

    def get_table(self):
        return self._cpt

    def get_node(self):
        return self._node

    def _get_parent_names(self):
        names = []
        for parent in self._node.get_parents():
            names.append(parent.get_name())

        return names

    def get_all_combination(self, l):
        """
        get_all_combination takes in a list of items, then returns all possible combinations of each item.
        :param l: list of items
        :return: list of combinations
        """
        combination_list = []
        for i in list(itertools.product(l, repeat=len(self._get_parent_names()))):
            combination_list.append(i)

        return combination_list

    def get_combination_probability(self, filtered_data, predict):
        """
        get_combination_probability will return the probability of the node's state given it's parents' states.
        :param filtered_data: DataFrame
        :param predict: state of Node
        :return: float
        """
        number_of_rows = filtered_data.shape[0]
        if number_of_rows == 0:
            return 0
        occurrence = len(filtered_data[filtered_data[self._node.get_name()] == predict])

        return occurrence / number_of_rows

    def update_cp_table(self, data, scale):
        """
        update_probability_table will up a DataFrame that will represent the Node's Conditional Probability Table
        (CP Table). This DataFrame will contain all of the probability of the Node's states given the parent's states.
        :param data: DataFrame
        :param scale: List of strings
        """
        cpt = self.create_data_frame_columns(scale)
        # When a Node has no parents then filtering based on parent states is not needed.
        if len(self._get_parent_names()) == 0:
            row = []
            for i in range(0, len(scale)):
                row.append(self.get_combination_probability(data, scale[i]))
            cpt.loc[0] = row

            self._cpt = cpt
        else:
            combo = self.get_all_combination(scale)

            for i in range(0, len(combo)):
                row = []
                row.extend(combo[i])
                for value in scale:
                    row.append(self.get_combination_probability(self.filter_data(combo[i]), value))
                cpt.loc[i] = row

            self._cpt = cpt

    def create_data_frame_columns(self, scale):
        """
        create_data_frame_columns will add all the node's parent's names as columns in the DataFrame. After the parent's
        each value in the scale will also be a column after the parents name.
        :param scale: list of Strings
        :return: DataFrame
        """
        columns = []

        if len(self._get_parent_names()) != 0:
            for parent in self._get_parent_names():
                columns.append(parent)

        columns.extend(scale)

        return pd.DataFrame(columns=columns)

    def filter_data(self, data, combination):
        """
        filter_data will filter the data frame based on the combination of the parents states.
        :param data: DataFrame
        :param combination: list of strings
        :return: Data Frame
        """
        df = data
        parent_names = self._get_parent_names()
        for i in range(0, len(parent_names)):
            temp_df = df[df[parent_names[i]] == combination[i]]
            if temp_df.shape[0] != 0:  # change to shape[0] >= 20
                df = temp_df
        return df
