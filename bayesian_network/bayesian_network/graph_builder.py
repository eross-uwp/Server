"""
___Authors___: Nate Braukhoff

__Purpose__: Building an AcyclicGraph from a CSV file.
"""

import random
import pandas as pd
from node import Node
from edge import Edge
from acyclic_graph import AcyclicGraph


class GraphBuilder:
    __POSTREQ = 'postreq'
    __PREREQ = 'prereqs'

    def __init__(self):
        self._nodes = []
        self._edges = []

    def get_parent_names(self, relation):
        """
        This method will pull out the class names that are in the logical statement and return a list of the names.
        resources: https://www.journaldev.com/23689/python-string-append
        :param relation: String
        :return: list
        """
        parents = []
        for i in range(0, len(relation)):
            if relation[i] == '{':
                name = []
                for j in range(i + 1, len(relation)):
                    if relation[j] != '#':
                        name.append(relation[j])
                    else:
                        i = j
                        break
                parents.append(''.join(name))
        return parents

    def get_node(self, name_of_node):
        """
        This method will return the node that has the given name.
        :param name_of_node: string
        :return: Node or None
        """
        for node in self._nodes:
            if node.get_name() == name_of_node:
                return node
        return None

    def new_build_nodes(self, node_names):
        for name in node_names:
            self._nodes.append(Node(name))

        return self

    def add_parents(self, relations):
        for i, row in relations.itterow():
            node_name = relations.at[i, self.__POSTREQ]
            relation = relations.at[i, self.__PREREQ]

            parent_name_list = self.get_parent_names(relation)
            node = self.get_node(node_name)

            for name in parent_name_list:
                node.add_parent(self.get_node(name))

        return self

    def add_children(self):
        for node in self._nodes:
            parents = node.get_parents()
            for parent in parents:
                parent.add_child(node)

        return self

    def new_build_edges(self):
        for node in self._nodes:
            children = node.get_children()
            for child in children:
                self._edges.append(Edge(node, child))

        return self

    def new_build_graph(self):
        return AcyclicGraph(self._nodes, self._edges)
