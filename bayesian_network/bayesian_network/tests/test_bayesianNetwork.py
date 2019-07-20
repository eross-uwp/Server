from unittest import TestCase
from node import Node
from edge import Edge
from acyclic_graph import AcyclicGraph
from bayesian_network import BayesianNetwork


class TestBayesianNetwork(TestCase):
    def test_get_graph(self):
        a = Node('A')
        b = Node('B')

        a.add_child(b)
        e = Edge(a, b)

        g = AcyclicGraph([a, b], [e])
        bn = BayesianNetwork(g, ['a', 'b'])

        self.assertTrue(g == bn.get_graph())

    def test_get_parameters(self):
        a = Node('A')
        b = Node('B')

        a.add_child(b)
        e = Edge(a, b)

        g = AcyclicGraph([a, b], [e])
        bn = BayesianNetwork(g, ['a', 'b'])

        self.assertTrue(['a', 'b'] == bn.get_parameters())

    def test_get_data_base(self):
        a = Node('A')
        b = Node('B')

        a.add_child(b)
        e = Edge(a, b)

        g = AcyclicGraph([a, b], [e])

        bn = BayesianNetwork(g, ['a', 'b'],
                             '/Users/natebraukhoff/Documents/Server/bayesian_network/test_data/Test_Data_AB.csv')

        print(bn.get_knowledge_base())


