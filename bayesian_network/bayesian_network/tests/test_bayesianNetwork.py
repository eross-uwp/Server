from unittest import TestCase
from node import Node
from edge import Edge
from acyclic_graph import AcyclicGraph
from bayesian_network import BayesianNetwork
from knowledge_base import KnowledgeBase
import pandas as pd


class TestBayesianNetwork(TestCase):
    def test_get_graph(self):
        kb = KnowledgeBase('..\\..\\test_data\\', '..\\..\\test_data\\doug_example.csv')
        bn = BayesianNetwork(kb)
        for node in bn.get_graph().get_nodes():
            print(node.get_name())

    def test_get_knowledge_base(self):
        kb = KnowledgeBase(None, '..\\..\\test_data\\doug_example.csv')
        bn = BayesianNetwork(kb)

        print(bn.get_knowledge_base().get_data())
        print(bn.get_knowledge_base().get_scale())

    def test_get_probability_of_node_state(self):
        self.fail()

    def test_get_node_relations(self):
        self.fail()

    def test_get_node_cp_table(self):
        self.fail()
