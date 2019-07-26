from unittest import TestCase
from node import Node
from conditional_probability_table import ConditionalProbabilityTable
import pandas as pd


class TestConditionalProbabilityTable(TestCase):
    def test_get_table(self):
        self.fail()

    def test_get_parent_list(self):
        test_data = pd.read_csv('..\\..\\test_data\\doug_example.csv')
        cpt = ConditionalProbabilityTable('Success', ['Work Hard', 'Smart'], test_data)
        print(cpt.get_parent_list())

    def test_get_owner(self):
        test_data = pd.read_csv('..\\..\\test_data\\doug_example.csv')
        cpt = ConditionalProbabilityTable('ClassA')
        self.assertTrue('ClassA' == cpt.get_owner_name())

    def test_get_all_combination(self):
        test_data = pd.read_csv('..\\..\\test_data\\doug_example.csv')
        cpt = ConditionalProbabilityTable('ClassA', ['Success', 'Hard Work', 'Smart'], test_data)
        cpt1 = ConditionalProbabilityTable('ClassA', ['Hard Work', 'Smart'], test_data)

        self.assertTrue(8 == len(cpt.get_all_combination(['T', 'F'])))
        print(cpt.get_all_combination(['T', 'F']))

        self.assertTrue(125 == len(cpt.get_all_combination(['A', 'B', 'C', 'D', 'F'])))
        print(cpt.get_all_combination(['A', 'B', 'C', 'D', 'F']))

        self.assertTrue(4 == len(cpt1.get_all_combination(['T', 'F'])))
        print(cpt1.get_all_combination(['T', 'F']))

        self.assertTrue(25 == len(cpt1.get_all_combination(['A', 'B', 'C', 'D', 'F'])))
        print(cpt1.get_all_combination(['A', 'B', 'C', 'D', 'F']))

    def test_filter_data(self):
        test_data = pd.read_csv('..\\..\\test_data\\doug_example.csv')
        cpt = ConditionalProbabilityTable('Success', ['Smart', 'Work Hard'], test_data)

        combo = ['T', 'T']

        print(cpt.filter_data(combo))

        combo = ['F', 'T']
        print(cpt.filter_data(combo))

        test_data1 = pd.read_csv('..\\..\\test_data\\missing_data.csv')
        cpt1 = ConditionalProbabilityTable('ClassC', ['ClassA', 'ClassB'], test_data1)
        print(cpt1.filter_data(['C', 'B']))

    def test_get_combination_probability(self):
        test_data = pd.read_csv('..\\..\\test_data\\doug_example.csv')
        cpt = ConditionalProbabilityTable('Success', ['Smart', 'Work Hard'], test_data)

        self.assertTrue(.8 == cpt.get_combination_probability(cpt.filter_data(['T', 'T']), 'T'))
        self.assertTrue(0 == cpt.get_combination_probability(cpt.filter_data(['T', 'I']), 'T'))

    def test_create_data_frame_columns(self):
        test_data = pd.read_csv('..\\..\\test_data\\doug_example.csv')
        cpt = ConditionalProbabilityTable('Success', ['Smart', 'Work Hard'], test_data)
        print(cpt.create_data_frame_columns(['T', 'F']))

        test_data = pd.read_csv('..\\..\\test_data\\doug_example.csv')
        cpt1 = ConditionalProbabilityTable('Smart', None, test_data)
        print(cpt1.create_data_frame_columns(['T', 'F']))

    def test_update_cp_table(self):
        test_data = pd.read_csv('..\\..\\test_data\\doug_example.csv')
        cpt = ConditionalProbabilityTable('Success', ['Smart', 'Work Hard'], test_data)

        cpt.update_cp_table(test_data, ['T', 'F'])
        print(cpt.get_table())

        cpt1 = ConditionalProbabilityTable('Work Hard', None, test_data)
        cpt1.update_cp_table(test_data, ['T', 'F'])
        print(cpt1.get_table())

        cpt2 = ConditionalProbabilityTable('Smart', None, test_data)
        cpt2.update_cp_table(test_data, ['T', 'F'])
        print(cpt2.get_table())


