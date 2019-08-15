from unittest import TestCase
from knowledge_base import KnowledgeBase
from conditional_probability_table_builder import CPTBuilder


class TestCPTBuilder(TestCase):
    def test_build(self):
        kb = KnowledgeBase('..\\..\\test_data\\doug_example.csv', '..\\..\\test_data\\doug_example.csv')
        builder = CPTBuilder('Success', ['Smart', 'Work Hard'], kb)

        kb1 = KnowledgeBase('..\\..\\test_data\\doug_example.csv', '..\\..\\test_data\\doug_example.csv')
        builder1 = CPTBuilder('Success', [], kb1)
        builder2 = CPTBuilder('Success', None, kb1)

        print(builder1.build())
        print(builder.build())
        print(builder2.build())

    def test__build_columns(self):
        kb = KnowledgeBase('..\\..\\test_data\\doug_example.csv', '..\\..\\test_data\\doug_example.csv')
        builder = CPTBuilder('Success', ['Smart', 'Work Hard'], kb)
        builder.build_columns()
        print(builder.get_table())

    def test_filter_data(self):
        kb = KnowledgeBase('..\\..\\test_data\\doug_example.csv', '..\\..\\test_data\\doug_example.csv')
        builder = CPTBuilder('Success', ['Smart', 'Work Hard'], kb)

        print(builder.filter_data(['T', 'T']))
        print(builder.filter_data(['F', 'T']))
        print(builder.filter_data(['A', 'B']))  # Combination is not apart of scale

        kb1 = KnowledgeBase('..\\..\\test_data\\missing_data.csv', '..\\..\\test_data\\missing_data.csv')
        builder1 = CPTBuilder('ClassC', ['ClassA', 'ClassB'], kb1)

        print(builder1.filter_data(['C', 'B']))

    def test_get_probability_of_combination(self):
        kb = KnowledgeBase('..\\..\\test_data\\doug_example.csv', '..\\..\\test_data\\doug_example.csv')
        builder = CPTBuilder('Success', ['Smart', 'Work Hard'], kb)

        self.assertTrue(.8 == builder.get_probability_of_combination(builder.filter_data(['T', 'T']), 'T'))
        self.assertTrue(0 == builder.get_probability_of_combination(builder.filter_data(['T', 'C']), 'T'))

    def test_get_all_combination(self):
        kb = KnowledgeBase('..\\..\\test_data\\doug_example.csv', '..\\..\\test_data\\doug_example.csv')
        builder = CPTBuilder('Success', ['Smart', 'Work Hard'], kb)

        self.assertTrue(4 == len(builder.get_all_combination()))
