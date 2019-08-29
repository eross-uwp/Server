"""
__Author__: Nate Braukhofff

__Purpose: The Acyclic Graph will be a directed graph that will not contain any cycles. This graph will be contain a
            list of Nodes and Edges. Users will be allowed to add and remove Nodes and Edges.
"""
from edge import Edge
from node import Node


class AcyclicGraph:
    def __init__(self, list_of_nodes, list_of_edges):
        self._nodes = list_of_nodes
        self._edges = list_of_edges

    def get_nodes(self):
        return self._nodes

    def get_edges(self):
        return self._edges

    def get_node(self, name_of_node):
        """
        Return a certain node in the graph. If the is not in the graph then get_node will return None
        :param name_of_node: string
        :return: Node or None
        """
        for node in self._nodes:
            if name_of_node == node.get_name():
                return node

        return None

    def get_edge(self, name_of_node):
        """
        Return a list of edges that is tied to a node. If the node is not in the graph then get_edge will return None
        :param name_of_node: String
        :return: list of edges
        """
        if self.get_node(name_of_node) is not None:
            edges = []
            for edge in self._edges:
                if edge.get_parent().get_name() == name_of_node or edge.get_child().get_name() == name_of_node:
                    edges.append(edge)
            return edges
        return None

    def add_node(self, parents, node):
        """
        add_node will add a the node to the graph a long with the edges that it will share with other nodes
        :param parents: list of parent nodes
        :param node: Node that is being added
        :return: void
        """
        for parent in parents:
            self._edges.append(Edge(parent, node))

        if node.get_children() is not None:
            for child in node.get_children():
                self._edges.append(Edge(node, child))
                self._nodes.append(child)

        self._nodes.append(node)

    def get_parents(self, name_of_node):
        """
        get_parents will find all the parent that a given node has. If a node doesn't have a parent then get_parents
        will return None
        :param name_of_node: String
        :return: list of Nodes or None
        """
        parents = []
        for edge in self._edges:
            if edge.get_child().get_name() == name_of_node:
                parents.append(edge.get_parent())

        if len(parents) == 0:
            return None
        else:
            return parents

    def add_edge(self, edge):
        """
        add_edge will add a new edge to the graph. The pre-condition for this method is both nodes need to exist in the
        graph. This method will also add children to the proper node.
        :param edge: Edge
        """
        self._edges.append(edge)

        for node in self._nodes:
            if node.get_name() == edge.get_parent().get_name():
                node.add_child(edge.get_child())

    def add_edges(self, list_of_edges):
        """
        add_edges will add multiple edges to the graph. The pre-condition for this method is all nodes need to exist in
        the graph. Children for each node will be added as well.
        :param list_of_edges: Edge list
        """
        for edge in list_of_edges:
            self.add_edge(edge)

    def remove_edge(self, name_of_parent, name_of_child):
        """
        remove_edge will remove the edge that is shared between the parent and the child. Once the edge is remove the
        method will also remove the child from the parent.
        :param name_of_parent: String
        :param name_of_child: String
        """
        for edge in self._edges:
            if edge.get_parent().get_name() == name_of_parent and edge.get_child().get_name() == name_of_child:
                self._edges.remove(edge)
                edge.get_parent().remove_child(name_of_child)

    def remove_node(self, name_of_node):
        """
        remove_node will remove the node from the graph. This method will also remove all edges that the node has with
        its parents and children.
        :param name_of_node: string
        """
        for node in self._nodes:
            if node.get_name() == name_of_node:

                # removing all edged with children
                for child in node.get_children():
                    self.remove_edge(node.get_name(), child.get_name())

                # removing all edges with parents
                parents = self.get_parents(node.get_name())
                for parent in parents:
                    self.remove_edge(parent.get_name(), node.get_name())
                    parent.remove_child(node.get_name())

                self._nodes.remove(node)
                break
