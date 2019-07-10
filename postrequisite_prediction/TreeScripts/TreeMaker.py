"""
___authors___: Austin FitzGerald and Evan Majerus

How to use: Create a TreeMaker object, giving it a file that contains a proper formatted logical prerequisite structure.
Call the process method on that object, giving it a class name, it will return the root node in the tree.
"""

import random
import pandas as pd
from TreeScripts.Node import Node


def __get_commas_for_split(operator, items):
    open_paren = 0
    commas_needed = 1
    char_iter = iter(items)
    for idx, char in enumerate(char_iter):
        if char == '(':
            open_paren += 1
            commas_needed += 1
        if char == ')':
            open_paren -= 1
        if idx > len(operator) and open_paren == 0:
            return commas_needed


class TreeMaker:
    __SINGLE_RELATIONSHIP = 'SINGLE'
    __AND_RELATIONSHIP = 'AND'
    __OR_RELATIONSHIP = 'OR'
    __POSTREQ = 'postreq'
    __PREREQ = 'prereqs'
    __READ_FILE = ''
    __OUTPUT_NAME = ''

    def __init__(self, file):
        self.__READ_FILE = pd.read_csv(file)
        self.__OUTPUT_NAME = file.split('\\')[2][:-4]

    def __create_trees(self, postreq, items):
        operator = items.split('(')[0]
        items = items[(len(operator) + 1):-1]  # remove beginning operator and last parenthesis

        # if and/or
        if operator == self.__AND_RELATIONSHIP or operator == self.__OR_RELATIONSHIP:
            commas_for_split = get_commas_for_split(operator, items)
            item_1 = items.split(',', commas_for_split)
            item_1 = ','.join(item_1[:commas_for_split])
            item_2 = items.split(',', commas_for_split)
            item_2 = ','.join(item_2[commas_for_split:])
            nodes = [Node('', operator), Node('', operator)]
            postreq.add_prereq(nodes[0])
            postreq.add_prereq(nodes[1])

            # if item is a class
            if item_1[0] == '{':
                nodes[0].set_grade(item_1.split('#')[1][:-1])
                nodes[0].set_name(item_1[item_1.find('{') + 1: item_1.find(
                    str('#') + nodes[0].get_grade() + str('}'))])
                self.create_trees(nodes[0], self.find_items(nodes[0].get_name()))

            if item_1.split('(')[0] == self.__AND_RELATIONSHIP or item_1.split('(')[0] == self.__OR_RELATIONSHIP or \
                    item_1.split('(')[0] == self.__SINGLE_RELATIONSHIP:
                nodes[0].set_name(random.randint(1, 101))
                self.create_trees(nodes[0], item_1)

            if item_2[0] == '{':
                nodes[1].set_grade(item_2.split('#')[1][:-1])
                nodes[1].set_name(item_2[item_2.find('{') + 1: item_2.find(
                    str('#') + nodes[1].get_grade() + str('}'))])
                self.create_trees(nodes[1], self.find_items(nodes[1].get_name()))
                return postreq

            if item_2.split('(')[0] == self.__AND_RELATIONSHIP or item_2.split('(')[0] == self.__OR_RELATIONSHIP or \
                    item_2.split('(')[0] == self.__SINGLE_RELATIONSHIP:
                nodes[1].set_name(random.randint(1, 101))
                self.create_trees(nodes[1], item_2)
                return postreq

        # if single relationship
        if operator == self.__SINGLE_RELATIONSHIP:
            a = Node('', self.__SINGLE_RELATIONSHIP)
            removed_operator = items.split('}')[0][1:]
            a.set_grade(removed_operator.split('#')[1])
            a.set_name(removed_operator.split('#')[0])
            postreq.add_prereq(a)
            return self.create_trees(a, self.find_items(a.get_name()))

    # returns the prereqs column cell that matches the given postreq
    def __find_items(self, class_name):
        for i, row in self.__READ_FILE.iterrows():
            if self.__READ_FILE.at[i, self.__POSTREQ] == class_name:
                return self.__READ_FILE.at[i, self.__PREREQ]
        return ''

    # get prereq tree for a given class
    def process(self, class_name):
        class_node = Node(class_name, 'SINGLE')
        items = self.find_items(class_name)
        self.create_trees(class_node, items)
        return class_node
