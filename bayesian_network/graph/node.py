"""
__Author__: Nate Braukhoff
"""


class Node:
    def __init__(self, name):
        self.name = name
        self.edges = []

    def get_name(self):
        return self.name

    def get_edges(self):
        return self.edges

    # get a sertian edge
    def get_edge(self, child):
        for edge in self.edges:
            if edge[1] == child:
                return edge[1]
        return None

    def add_edge(self, edge):
        self.edges.append(edge)

