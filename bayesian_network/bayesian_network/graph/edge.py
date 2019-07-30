"""
__Author__: Nate Braukhofff

__Purpose: The Edge will contain the following:
        A pair a nodes that will contain the parent and the child
        Data that will be sent from the parent to the child
"""


class Edge:
    def __init__(self, node1, node2):
        self._node1 = node1
        self._node2 = node2


