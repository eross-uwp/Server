"""
__Author__: Nate Braukhoff

__Purpose__: The Bayesian Network will consist of a graph and a list of Parameters. Will be able to calculate any
             probability of a Node in the Graph. The results will be outputted to the terminal.
"""
from knowledge_base import KnowledgeBase


class BayesianNetwork:

    def __init__(self, graph, parameters_set, knowledge_base=None):
        if knowledge_base is None:
            knowledge_base = KnowledgeBase(None, None)
        self._graph = graph
        self._parameters_set = parameters_set
        self._knowledge_base = knowledge_base

    def get_graph(self):
        return self._graph

    def get_parameters(self):
        return self._parameters_set

    def get_knowledge_base(self):
        return self._knowledge_base

    def get_probability_for_node(self, name_of_start, grade, name_of_finish):
        return 1
