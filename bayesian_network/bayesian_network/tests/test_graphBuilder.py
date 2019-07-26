from unittest import TestCase
from graph_builder import GraphBuilder
from node import Node
import pandas as pd
from acyclic_graph import AcyclicGraph


class TestGraphBuilder(TestCase):
    def test_build_graph(self):
        relations = pd.read_csv('..\\..\\ExcelFiles\\se.csv')
        node_names = list(relations['postreq'])
        print(node_names)

        gb = GraphBuilder(node_names, relations)
        g = gb.build_graph()
        print(g.get_node('Calculus and Analytic Geometry II').get_parents()[0].get_name())

    def test_get_parent_names(self):
        gb = GraphBuilder([], None)
        parents = ['Calculus and Analytic Geometry II#c-']
        print(gb.get_parent_names('SINGLE({Calculus and Analytic Geometry II#c-})'))
        print(gb.get_parent_names('AND(AND({Object-Oriented Programming and Data Structures II#c-},'
                                  '{Object Oriented Analysis and Design#c-}),OR({Introduction to Microprocessors#c-},'
                                  '{Computer Architecture/Operating Systems#c-}))'))
        # self.assertTrue(parents == gb.get_parent_names('SINGLE({Calculus and Analytic Geometry II#c-})'))
