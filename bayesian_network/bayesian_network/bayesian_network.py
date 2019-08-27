"""
__Author__: Nate Braukhoff

__Purpose__: The Bayesian Network will consist of a graph and a list of Parameters. Will be able to calculate any
             probability of a Node in the Graph. The results will be outputted to the terminal.
"""
from conditional_probability_table import ConditionalProbabilityTable
from conditional_probability_table_builder import CPTBuilder


class BayesianNetwork:

    def __init__(self, knowledge_base, graph):
        self._graph = graph
        self._kb = knowledge_base
        self._cpt_dictionary = dict()
        '''
        for node in graph.get_nodes():
            builder = CPTBuilder(node.get_name(), node.get_parent_names())
            table = builder.build(knowledge_base.get_data(), knowledge_base.get_scale())
            self._cpt_dictionary.update({node.get_name(): ConditionalProbabilityTable(table)})
        '''
    def get_graph(self):
        return self._graph

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

