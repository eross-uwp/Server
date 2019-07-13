"""
__Author__: Nate Braukhoff

__Purpose__: The Bayesian Network will consist of a graph and a list of Parameters. Will be able to calculate any
             probability of a Node in the Graph. The results will be outputted to the terminal.
"""


class BayesianNetwork:

    def __init__(self, graph, parameters_set):
        self.graph = graph
        self.parameters_set = parameters_set
