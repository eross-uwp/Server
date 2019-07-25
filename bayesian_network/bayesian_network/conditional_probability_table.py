import itertools

import pandas as pd


class ConditionalProbabilityTable:
    def __init__(self, data):
        self._data = data
        self._cpt = pd.DataFrame()

    # todo implement all methods from the Node class
    def get_table(self):
        return self._cpt

    def get_all_combination(self, l, number_of_parents):
        """
        get_all_combination takes in a list of items, then returns all possible combinations of each item.
        :param number_of_parents: int
        :param l: list of items
        :return: list of combinations
        """
        combination_list = []
        for i in list(itertools.product(l, repeat=number_of_parents)):
            combination_list.append(i)

        return combination_list

    def get_combination_probability(self, filtered_data, predict, name_of_node):
        """
        get_combination_probability will return the probability of the node's state given it's parents' states.
        :param name_of_node: string
        :param filtered_data: DataFrame
        :param predict: state of Node
        :return: float
        """
        number_of_rows = filtered_data.shape[0]
        if number_of_rows == 0:
            return 0
        occurrence = len(filtered_data[filtered_data[name_of_node] == predict])

        return occurrence / number_of_rows

    def update_cp_table(self, data, scale, number_of_parents, name):
        """
        update_probability_table will up a DataFrame that will represent the Node's Conditional Probability Table
        (CP Table). This DataFrame will contain all of the probability of the Node's states given the parent's states.
        :param name: string
        :param number_of_parents: integer
        :param data: DataFrame
        :param scale: List of strings
        """
        cpt = self.create_data_frame_columns(scale)
        # When a Node has no parents then filtering based on parent states is not needed.
        if len(number_of_parents) == 0:
            row = []
            for i in range(0, len(scale)):
                row.append(self.get_combination_probability(data, scale[i], name))
            cpt.loc[0] = row

            self._cpt = cpt
        else:
            combo = self.get_all_combination(scale, number_of_parents)

            for i in range(0, len(combo)):
                row = []
                row.extend(combo[i])
                for value in scale:
                    row.append(self.get_combination_probability(self.filter_data(data, combo[i]), value, name))
                cpt.loc[i] = row

            self._cpt = cpt

    def create_data_frame_columns(self, scale, parents):
        """
        create_data_frame_columns will add all the node's parent's names as columns in the DataFrame. After the parent's
        each value in the scale will also be a column after the parents name.
        :param parents: list
        :param scale: list of Strings
        :return: DataFrame
        """
        columns = []

        if len(parents) != 0:
            for parent in parents:
                columns.append(parent.get_name())

        columns.extend(scale)

        return pd.DataFrame(columns=columns)

    def filter_data(self, data, combination, parents):
        """
        filter_data will filter the data frame based on the combination of the parents states.
        :param parents:
        :param data: Data Frame. Pre condition: Node must have at least one parent
        :param combination: list of strings
        :return: Data Frame
        """
        df = data

        for i in range(0, len(parents)):
            df = df[df[parents[i].get_name()] == combination[i]]
        return df
