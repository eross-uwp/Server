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

