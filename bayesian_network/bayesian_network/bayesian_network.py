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

    # Todo: Send node it's parents information
    def update_node_table(self, node_name):
        node = self._graph.get_nod(node_name)
        parent_list = []

        for parent in node.get_parents():
            parent_list.append(parent.get_name())
        parent_list.append(node.get_name())

        node.update_probability_table(self._knowledge_base.get_class_data(parent_list))
