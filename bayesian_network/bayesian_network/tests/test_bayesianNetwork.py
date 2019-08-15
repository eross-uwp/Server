from unittest import TestCase
from node import Node
from edge import Edge
from acyclic_graph import AcyclicGraph
from bayesian_network import BayesianNetwork
from knowledge_base import KnowledgeBase
import pandas as pd


class TestBayesianNetwork(TestCase):
    def test_BayesianNetwork(self):
        node_a = Node('Smart')
        node_b = Node('Work Hard')
        node_c = Node('Success')

        node_a.add_child(node_c)
        node_b.add_child(node_c)

        node_c.add_parent(node_a)
        node_c.add_parent(node_b)

        g = AcyclicGraph([node_a, node_b, node_c], [])
        kb = KnowledgeBase('..\\..\\test_data\\doug_example.csv', '..\\..\\test_data\\doug_example.csv')

        bn = BayesianNetwork(kb, g)
        print()

    def test_get_graph(self):
        kb = KnowledgeBase('..\\..\\test_data\\', '..\\..\\test_data\\doug_example.csv')
        bn = BayesianNetwork(kb)
        for node in bn.get_graph().get_nodes():
            print(node.get_name())

    def test_get_probability_of_node_state(self):
        self.fail()

    def test_get_node_relations(self):
        self.fail()

    def test_get_node_cp_table(self):
        self.fail()
