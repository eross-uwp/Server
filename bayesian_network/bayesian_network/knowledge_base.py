""""
__Author__: Nate Braukhoff

__Purpose__: The Knowledge Base will store all the grades for all students for each class
"""
import pandas as pd


class KnowledgeBase:
    def __init__(self, relations_file_path=None, class_data_file_path=None):
        if relations_file_path is None:
            self._relations = pd.DataFrame()
        else:
            self._relations = pd.read_csv(relations_file_path)

        if class_data_file_path is None:
            self._data = pd.DataFrame()
        else:
            self._data = pd.read_csv(class_data_file_path)

    def get_data(self):
        return self._data

    def get_relations(self):
        return self._relations

    def add_data(self, data_file_path):
        """
        add_data will add columns to the bayesian network's DataFrame
        :param data_file_path: string
        """
        df = pd.read_csv(data_file_path)
        self._data = pd.concat([self._data, df], axis=1)

    def update_data(self, data_file_path):
        """
        update_data updates existing columns in the bayesian network's DataFrame
        :param data_file_path: string
        """
        df = pd.read_csv(data_file_path)
        self._data.update(df)

    def update_relation(self, relation_file_path):
        return None

    def add_relation(self, relation_file_path):
        return None

    def get_class_data(self, class_list):
        return None