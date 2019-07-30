""""
__Author__: Nate Braukhoff

__Purpose__: The Knowledge Base will store all the grades for all students for each class
"""
import pandas as pd


class KnowledgeBase:
    def __init__(self, relations_file_path, class_data_file_path):
        self._relations = pd.read_csv(relations_file_path)
        self._data = pd.read_csv(class_data_file_path)

        # https: // stackoverflow.com / questions / 26977076 / pandas - unique - values - multiple - columns
        self._scale = pd.unique(self._data[self._data.columns].values.ravel('k'))

    def get_data(self):
        return self._data

    def get_relations(self):
        return self._relations

    def get_relation(self, class_name):
        """
        get_relation will return the relation that the Node has with another Node. If no relation exists then the
        method will return None
        :param class_name: String
        :return: String or None
        """
        # Todo need to implement this method when Node relation comes in play
        return self

    def get_class_data(self, class_name_list):
        """
        get_class_data will return a Data Frame that contains all data for each class in the class_name_list. If a class
        doesn't exist in _data will not return anything for that class. If all classes don't exist in _data then this
        method will return None
        :param class_name_list: List of Strings
        :return: Data Frame or None
        """
        valid_name = []
        for name in class_name_list:
            if name in self._data.columns:
                valid_name.append(name)
        if len(valid_name) != 0:
            return self._data[valid_name]
        else:
            return None

    def get_scale(self):
        return self._scale

    def add_data(self, data_file_path):
        """
        add_data will add columns to the bayesian network's DataFrame
        :param data_file_path: string
        """
        df = pd.read_csv(data_file_path)
        self._data = pd.concat([self._data, df], axis=1)
        self.update_scale()

    def update_data(self, data_file_path):
        """
        update_data updates existing columns in the bayesian network's DataFrame
        :param data_file_path: string
        """
        df = pd.read_csv(data_file_path)
        self._data.update(df)
        self.update_scale()

    def update_relation(self, relation_file_path):
        """
        update an existing relation between Nodes
        :param relation_file_path: String
        """
        # Todo need to implement
        return self

    def update_scale(self):
        self._scale = pd.unique(self._data[self._data.columns].values.ravel('k'))

    def add_relation(self, relation_file_path):
        """
        Adds relations between Nodes
        :param relation_file_path: String
        """
        # Todo need to implement
        return self

