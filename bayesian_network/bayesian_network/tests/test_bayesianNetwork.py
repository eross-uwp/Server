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

        print(bn.get_data_base())

    def test_add_data(self):
        a = Node('A')
        b = Node('B')
        c = Node('C')

        a.add_child(b)
        e1 = Edge(a, b)
        e2 = Edge(a, c)

        g = AcyclicGraph([a, b], [e1, e2])

        bn = BayesianNetwork(g, ['a', 'b'],
                             '/Users/natebraukhoff/Documents/Server/bayesian_network/test_data/Test_Data_AB.csv')
        bn.add_data('/Users/natebraukhoff/Documents/Server/bayesian_network/test_data/Test_Data_C.csv')

        print(bn.get_data_base())

    def test_update_data(self):
        a = Node('A')
        b = Node('B')

        a.add_child(b)
        e = Edge(a, b)

        g = AcyclicGraph([a, b], [e])

        bn = BayesianNetwork(g, ['a', 'b'],
                             '/Users/natebraukhoff/Documents/Server/bayesian_network/test_data/Test_Data_AB.csv')

        print(bn.get_data_base())

        bn.update_data('/Users/natebraukhoff/Documents/Server/bayesian_network/test_data/test_data_A_update.csv')

        print(bn.get_data_base())

    def test_get_probability_for_serial_relationship(self):
        smart = Node('Smart')
        work_hard = Node('Work Hard')
        success = Node('Success')

        smart.add_child(work_hard)
        work_hard.add_child(success)

        sw = Edge(smart, work)
        wh = Edge()
