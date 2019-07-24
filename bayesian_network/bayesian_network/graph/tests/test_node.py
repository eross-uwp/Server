import itertools
from unittest import TestCase
from node import Node
from itertools import combinations
from itertools import combinations_with_replacement
import pandas as pd


class TestNode(TestCase):
    def test_get_name(self):
        a = Node('ClassA', None)
        self.assertEqual('ClassA', a.get_name())

    def test_get_children(self):
        a = Node('ClassA', [Node('B'), Node('C'), Node('D')])
        children = [Node('B'), Node('C'), Node('D')]

        self.assertTrue(children[0].get_name() == a.get_children()[0].get_name()
                        and children[1].get_name() == a.get_children()[1].get_name()
                        and children[2].get_name() == a.get_children()[2].get_name())

        self.assertTrue(3 == len(a.get_children()))

    def test_get_child(self):
        a = Node('ClassA', [Node('B'), Node('C'), Node('D')])
        b = Node('ClassB')

        self.assertTrue('B' == a.get_child('B').get_name())

        self.assertTrue(a.get_child('Z') is None)

        self.assertTrue(b.get_child('A') is None)

    def test_add_child(self):
        a = Node('ClassA', [Node('B'), Node('C'), Node('D')])
        b = Node('ClassB')

        a.add_child(Node('Z'))
        b.add_child(Node('Z'))

        self.assertTrue('Z' == a.get_children()[3].get_name())
        self.assertTrue(4 == len(a.get_children()))

        self.assertTrue('Z' == b.get_children()[0].get_name())
        self.assertTrue(1 == len(b.get_children()))

        b.add_child(Node('X'))

        self.assertTrue('X' == b.get_children()[1].get_name())
        self.assertTrue(2 == len(b.get_children()))

    def test_get_all_combination(self):
        a = Node('A')
        b = Node('B')
        c = Node('C')

        a.add_parent(b)
        a.add_parent(c)

        self.assertTrue(4 == len(a.get_all_combination(['T', 'F'])))
        print(a.get_all_combination(['T', 'F']))

        self.assertTrue(25 == len(a.get_all_combination(['A', 'B', 'C', 'D', 'F'])))
        print(a.get_all_combination(['A', 'B', 'C', 'D', 'F']))

    def test_get_combination_probability(self):
        a = Node('Smart')
        b = Node('Work Hard')
        c = Node('Success')

        c.add_parent(a)
        c.add_parent(b)

        test_data = pd.read_csv('..\\..\\..\\test_data\\doug_example.csv')
        self.assertTrue(.8 == c.get_combination_probability(c.filter_data(test_data, ['T', 'T']), 'T'))
        self.assertTrue(0 == c.get_combination_probability(c.filter_data(test_data, ['T', 'I']), 'T'))

    def test_remove_child(self):
        a = Node('A')
        b = Node('B')
        c = Node('C')

        a.get_child(b)
        a.add_child(c)

        a.remove_child('B')

        self.assertTrue(1 == len(a.get_children()))
        self.assertTrue(c == a.get_children()[0])

    def test_add_parent(self):
        a = Node('A')
        b = Node('B')
        c = Node('C')

        a.add_parent(a)

        self.assertTrue(a == a.get_parents()[0])

    def test_update_probability_table(self):
        a = Node('Smart')
        b = Node('Work Hard')
        c = Node('Success')

        c.add_parent(a)
        c.add_parent(b)

        test_data = pd.read_csv('..\\..\\..\\test_data\\doug_example.csv')

        c.update_cp_table(test_data, ['T', 'F'])
        print(c.get_cp_table())

        a.update_cp_table(test_data, ['T', 'F'])
        print(a.get_cp_table())

        b.update_cp_table(test_data, ['T', 'F'])
        print(b.get_cp_table())

    def test_filter_data(self):
        a = Node('Smart')
        b = Node('Work Hard')
        c = Node('Success')
        test_data = pd.read_csv('..\\..\\..\\test_data\\doug_example.csv')

        c.add_parent(a)
        c.add_parent(b)

        combo = ['T', 'T']

        print(c.filter_data(test_data, combo))

        combo = ['F', 'T']
        print(c.filter_data(test_data, combo))
