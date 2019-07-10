"""
__Author__: Nate Braukhoff
"""


class Edge:
    def __init__(self, parent, child):
        self.edge = [parent, child]

    def get_source(self):
        return self.edge[0]

    def get_destination(self):
        return self.edge[1]