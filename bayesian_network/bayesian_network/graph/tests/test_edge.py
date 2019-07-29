from unittest import TestCase
from edge import Edge
from node import Node


class TestEdge(TestCase):
    def test_get_parent(self):
        a = Node('A')
        b = Node('B')

        e = Edge(a, b)

        self.assertTrue('A' == e.get_parent().get_name())

    def test_get_destination(self):
        a = Node('A')
        b = Node('B')

        e = Edge(a, b)

        self.assertTrue('B' == e.get_child().get_name())
