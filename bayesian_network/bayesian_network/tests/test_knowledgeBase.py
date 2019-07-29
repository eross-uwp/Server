from unittest import TestCase
from knowledge_base import KnowledgeBase


class TestKnowledgeBase(TestCase):
    def test_create_scale(self):
        kb = KnowledgeBase(None, '/Users/natebraukhoff/Documents/Server/bayesian_network/test_data/doug_example.csv')
        print(kb.get_scale())

    def test_get_data(self):
        kb = KnowledgeBase(None, '..\\..\\test_data\\Test_Data_AB.csv')
        print(kb.get_data())

    def test_add_data(self):
        kb = KnowledgeBase(None, '..\\..\\test_data\\Test_Data_AB.csv')
        print(kb.get_data())
        print()
        kb.add_data('..\\..\\test_data\\Test_Data_C.csv')
        print(kb.get_data())

    def test_update_data(self):
        kb = KnowledgeBase(None, '..\\..\\test_data\\Test_Data_AB.csv')
        print(kb.get_data())
        print()
        kb.update_data('..\\..\\test_data\\test_data_A_update.csv')
        print(kb.get_data())

    def test_get_class_data(self):
        kb = KnowledgeBase(None, '..\\..\\test_data\\Test_Data_AB.csv')
        kb.add_data('..\\..\\test_data\\Test_Data_C.csv')
        print(kb.get_data())
        print()

        print(kb.get_class_data(['A', 'C']))
        print()
        print(kb.get_class_data(['B']))
        print(kb.get_class_data(['Z']))
        print(kb.get_class_data(['A', 'B', 'Z']))

