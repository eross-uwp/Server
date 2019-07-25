from unittest import TestCase
from node import Node


class TestConditionalProbabilityTable(TestCase):
    def test_get_table(self):
        self.fail()

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

    def test_update_cp_table(self):
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

    def test_create_data_frame_columns(self):
        self.fail()

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
