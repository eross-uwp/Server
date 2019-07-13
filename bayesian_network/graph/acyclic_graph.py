"""
__Author__: Nate Braukhofff

__Purpose: The Acyclic Graph will be a directed graph that will not contain any cycles. This graph will be contain a
            list of Nodes and Edges. Users will be allowed to add and remove Nodes and Edges.
"""


class AcyclicGraph:

    def __init__(self, list_of_nodes, list_of_edges):
        self.nodes = list_of_nodes
        self.edges = list_of_edges
