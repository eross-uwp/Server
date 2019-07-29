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

    def __init__(self, node_names, relations):
        """
        The constructor will create a graph builder.
        :param node_names: list
        :param relations: DataFrame
        """
        nodes = []
        for name in node_names:
            nodes.append(Node(name))
        self._nodes = nodes
        self._relations = relations

    def build_graph(self):
        """
        This method will create a list of nodes and a list of edges. With the two list the method will create and return
        an acyclic graph
        :return: AcyclicGraph
        """
        self._nodes = self.build_nodes()

        edges = []
        for node in self._nodes:
            edges.extend(self.build_edges(node))

        return AcyclicGraph(self._nodes, edges)

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

    def build_edges(self, node):
        """
        This method will iterate through nodes child list and create an edge for each child.
        :param node: Node
        :return: list
        """
        if node.get_children() is not None:
            edges = []
            for child in node.get_children():
                edges.append(Edge(node, child))
            return edges

    def build_nodes(self):
        """
        This method will created each node in the graph. Then it will add the nodes' children and parents for each node.
        Finally create edges that are in the graph.
        :return:
        """
        nodes = []
        for i, row in self._relations.iterrows():
            name = self._relations.at[i, self.__POSTREQ]
            relation = self._relations.at[i, self.__PREREQ]

            name_of_parents = self.get_parent_names(relation)
            node = self.get_node(name)

            for p_name in name_of_parents:
                node.add_parent(self.get_node(p_name))
                parent = self.get_node(p_name)

                if parent is None:
                    nodes.append(Node(p_name, [node]))
                else:
                    parent.add_child(node)
                    nodes.append(node)
        return nodes
