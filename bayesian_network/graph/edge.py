"""
__Author__: Nate Braukhofff

__Purpose: The Edge will contain the following:
        A pair a nodes that will contain the parent and the child
        Data that will be sent from the parent to the child
"""


class Edge:
    def __init__(self, parent, child):
        self.edge = [parent, child]

    def get_source(self):
        return self.edge[0]

    def get_destination(self):
        return self.edge[1]