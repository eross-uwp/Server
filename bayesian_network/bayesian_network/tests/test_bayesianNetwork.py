from unittest import TestCase

from graph_builder import GraphBuilder
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

        # test with our data set
        _data_file_path = '..\\..\\ExcelFiles\\courses_and_grades.csv'
        _relations_file_path = '..\\..\\..\\Data\\combined_course_structure.csv'

        knowledge_base = KnowledgeBase(_relations_file_path, _data_file_path)

        builder = GraphBuilder()
        builder = builder.build_nodes(list(knowledge_base.get_data().columns))
        builder = builder.add_parents(knowledge_base.get_relations())
        builder = builder.add_children()
        builder = builder.build_edges()

        graph = builder.build_graph()
        bayes_net = BayesianNetwork(knowledge_base, graph)
        print()