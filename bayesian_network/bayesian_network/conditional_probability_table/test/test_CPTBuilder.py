from unittest import TestCase

from conditional_probability_table_builder import CPTBuilder
from knowledge_base import KnowledgeBase


class TestCPTBuilder(TestCase):
    def test_CPTBuilder(self):
        kb = KnowledgeBase('..\\..\\..\\test_data\\doug_example.csv', '..\\..\\..\\test_data\\doug_example.csv')
        data = kb.get_query(['Success', 'Smart', 'Work Hard'])

        builder = CPTBuilder(data, kb.get_scale())
        print()

    def test_make_columns(self):
        kb = KnowledgeBase('..\\..\\..\\test_data\\doug_example.csv', '..\\..\\..\\test_data\\doug_example.csv')
        data = kb.get_query(['Success', 'Smart', 'Work Hard'])

        builder = CPTBuilder(data, kb.get_scale())
        print(builder._make_columns())

    def test_apply_combination_filter(self):
        kb = KnowledgeBase('..\\..\\..\\test_data\\doug_example.csv', '..\\..\\..\\test_data\\doug_example.csv')
        data = kb.get_query(['Success', 'Smart', 'Work Hard'])

        builder = CPTBuilder(data, kb.get_scale())
        print(builder._apply_combination_filter(['T', 'T']))
        print(builder._apply_combination_filter(['F', 'T']))
        print(builder._apply_combination_filter(['A', 'B']))  # Combination is not apart of scale

    def test_calculate_probability(self):
        kb = KnowledgeBase('..\\..\\..\\test_data\\doug_example.csv', '..\\..\\..\\test_data\\doug_example.csv')
        data = kb.get_query(['Success', 'Smart', 'Work Hard'])

        builder = CPTBuilder(data, kb.get_scale())

        fd = list(builder._apply_combination_filter(['T', 'T']))
        self.assertTrue(.8 == builder._calculate_probability(fd, 'T'))

        fd = list(builder._apply_combination_filter(['T', 'C']))
        self.assertTrue(0 == builder._calculate_probability(fd, 'T'))

        kb1 = KnowledgeBase('..\\..\\..\\test_data\\doug_example.csv', '..\\..\\..\\test_data\\doug_example.csv')
        data = kb1.get_query(['Work Hard'])

        builder1 = CPTBuilder(data, kb.get_scale())
        self.assertTrue((11/24) == builder1._calculate_probability(list(data.values), 'T'))
        self.assertTrue((13/24) == builder1._calculate_probability(list(data.values), 'F'))

    def test_build_with_no_parents(self):
        kb = KnowledgeBase('..\\..\\..\\test_data\\doug_example.csv', '..\\..\\..\\test_data\\doug_example.csv')
        data = kb.get_query(['Work Hard'])
        builder = CPTBuilder(data, kb.get_scale())

        print(builder.build_with_no_parents())

        kb1 = KnowledgeBase('..\\..\\..\\test_data\\doug_example.csv', '..\\..\\..\\test_data\\doug_example.csv')
        data = kb1.get_query(['Smart'])
        builder1 = CPTBuilder(data, kb.get_scale())

        print(builder1.build_with_no_parents())

