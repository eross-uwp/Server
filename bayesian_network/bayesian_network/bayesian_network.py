"""
__Author__: Nate Braukhoff

__Purpose__: The Bayesian Network will consist of a graph and a list of Parameters. Will be able to calculate any
             probability of a Node in the Graph. The results will be outputted to the terminal.
"""
import pandas as pd
import numpy as np


class BayesianNetwork:

    def __init__(self, graph, parameters_set, data_file_path=None):
        if data_file_path is None:
            self._data_base = pd.DataFrame()
        else:
            self._data_base = pd.read_csv(data_file_path)

        self._graph = graph
        self._parameters_set = parameters_set

    def get_graph(self):
        return self._graph

    def get_parameters(self):
        return self._parameters_set

    def get_data_base(self):
        return self._data_base

    def add_data(self, data_file_path):
        """
        add_data will add columns to the bayesian network's DataFrame
        :param data_file_path: string
        """
        df = pd.read_csv(data_file_path)
        self._data_base = pd.concat([self._data_base, df], axis=1)

    def update_data(self, data_file_path):
        """
        update_data updates existing columns in the bayesian network's DataFrame
        :param data_file_path: string
        """
        df = pd.read_csv(data_file_path)
        self._data_base.update(df)


    def get_probability_for_node(self, name_of_start, grade, name_of_finish):
        return 1
