"""
__Author__: Nate Braukhoff

__Purpose__: The Bayesian Network will consist of a graph and a list of Parameters. Will be able to calculate any
             probability of a Node in the Graph. The results will be outputted to the terminal.
"""
from knowledge_base import KnowledgeBase
import pandas as pd
from acyclic_graph import AcyclicGraph
from node import Node


class BayesianNetwork:

    def __init__(self, knowledge_base=None):
        if knowledge_base is None:
            knowledge_base = KnowledgeBase(None, None)
        self._knowledge_base = knowledge_base

        node_names = knowledge_base.get_data().columns
        self._graph = AcyclicGraph(node_names, knowledge_base.get_relations())
        
        for node in self._graph.get_nodes():
            node.get_cp_table().update_cp_table(self._knowledge_base.get_data() ,self._knowledge_base.get_scale())

    def get_graph(self):
        return self._graph

    def get_knowledge_base(self):
        return self._knowledge_base

    def get_probability_of_node_state(self, name_of_node, state):
        """
        This method will get a probability of a node's state bases the current state of it's parents.
        :param name_of_node: string
        :param state: string
        :return: float
        """
        return .42

    def get_node_cp_table(self, name_of_node):
        """
        This method will return a DataFrame that will contain all probabilities for each state of the node
        :param name_of_node: string
        :return: DataFrame
        """
        return self._graph.get_node(name_of_node).get_cp_table()

    def create_cpt_tables(self):
        """
        This method will iterate through all nodes in the graph and create a CPT table for everyone.
        :return:
        """
        for node in self._graph.get_nodes():
            node.get_cp_table().update_cp_table(self._knowledge_base.get_data() ,self._knowledge_base.get_scale())
