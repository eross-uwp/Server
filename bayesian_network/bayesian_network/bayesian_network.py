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

        # make graph
        # get all node names
        # create nodes and add children and parents to each node.
        node_list = []
        names = knowledge_base.get_data().columns
        for i in range(0, len(names)):
            node_list.append(Node(names[i]))

        self._graph = AcyclicGraph(None, None)

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
        return .420

    def get_node_relations(self, name_of_node):
        """
        This method will return all relations that the node has with it's children.
        :param name_of_node: string
        :return: list
        """
        return ['Hello', 'World']

    def get_node_cp_table(self, name_of_node):
        """
        This method will return a DataFrame that will contain all probabilities for each state of the node
        :param name_of_node: string
        :return: DataFrame
        """
        return 'DataFrame'
