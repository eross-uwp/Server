"""
___authors___: Austin FitzGerald, Evan Majerus, Nate Braukhoff, Zhiwei Yang

How to use: Create a TreeMaker object, giving it a file that contains a proper formatted logical prerequisite structure.
Call the process method on that object, giving it a class name, it will return the root node in the tree.

Note: The words operator and relationship are used interchangeably throughout the code. An item is what is contained
inside a logical expression. There are exactly 2 items per AND/OR logical expression, and 1 item for a SINGLE logical
expression. Items can either be classes or other logical expressions.
"""

import random
import pandas as pd
from TreeScripts.Node import Node


class TreeMaker:
    __SINGLE_RELATIONSHIP = 'SINGLE'
    __AND_RELATIONSHIP = 'AND'
    __OR_RELATIONSHIP = 'OR'
    __POSTREQ = 'postreq'
    __PREREQ = 'prereqs'
    __READ_FILE = ''
    __OUTPUT_NAME = ''
    virtual_node_name = 0

    def __init__(self, file):
        """
        The constructor for a TreeMaker object. Takes in a string that contains the filepath for the course structure
        logical list. Needs a "postreq" column which contains class names that act as postrequisites. Needs a "prereqs"
        column which contains the logical expression for the prereqs.
        :param file:
        """
        self.__READ_FILE = pd.read_csv(file)
        self.__OUTPUT_NAME = file.rpartition('\\')[:-1][:-4]

    def __create_trees(self, postreq, prereqs):
        """
        A recursive descent parser that will generate a prerequisite structure tree for a given class. Each class node
        has a maximum of 2 children, so virtual nodes are used for classes that require more than one relationship type.
        :param postreq: The postreq class node which will act as the head node in the tree.
        :param prereqs: The list of prerequisites for the given postrequisite node. Taken from the csv, contains a parser
        expression grammar that is notated by logical expressions (AND, OR, SINGLE). Items are the prereq class names.
        :return: The postrequisite, for which all classes in the tree will act as at one point (recursively).
        """
        operator = prereqs.split('(')[0]
        prereqs = prereqs[(len(operator) + 1):-1]  # remove beginning operator and last parenthesis

        # if AND/OR relationship
        if operator == self.__AND_RELATIONSHIP or operator == self.__OR_RELATIONSHIP:
            commas_for_split = self.__get_commas_for_split(operator, prereqs)
            item_1 = prereqs.split(',', commas_for_split)
            item_1 = ','.join(item_1[:commas_for_split])
            item_2 = prereqs.split(',', commas_for_split)
            item_2 = ','.join(item_2[commas_for_split:])
            nodes = [Node('', operator), Node('', operator)]
            postreq.add_prereq(nodes[0])
            postreq.add_prereq(nodes[1])

            # if item 1 is a class, set grade and name for node 1, call create_trees with item 1 as postreq
            if item_1[0] == '{':
                nodes[0].set_grade(item_1.split('#')[1][:-1])
                nodes[0].set_name(item_1[item_1.find('{') + 1: item_1.find(
                    str('#') + nodes[0].get_grade() + str('}'))])
                self.__create_trees(nodes[0], self.__find_items(nodes[0].get_name()))
            # if item 1 is an operator, set name for node 1 to the incrementing virtual_node_name, call create_trees
            # with node 1 as the postreq and item 1 as the prereqs list
            if item_1.split('(')[0] == self.__AND_RELATIONSHIP or item_1.split('(')[0] == self.__OR_RELATIONSHIP or \
                    item_1.split('(')[0] == self.__SINGLE_RELATIONSHIP:
                nodes[0].set_name(self.virtual_node_name)
                nodes[0].set_virtual(nodes[0].VIRTUAL_TYPE)
                self.virtual_node_name += 1
                self.__create_trees(nodes[0], item_1)

            # if item 2 is a class, set name and grade for node 1, call create_trees with item 2 as posteq, and return
            # the original postreq, because we are done
            if item_2[0] == '{':
                nodes[1].set_grade(item_2.split('#')[1][:-1])
                nodes[1].set_name(item_2[item_2.find('{') + 1: item_2.find(
                    str('#') + nodes[1].get_grade() + str('}'))])
                self.__create_trees(nodes[1], self.__find_items(nodes[1].get_name()))
                return postreq

            # if item 2 is an operator, set name for node 2 to the incrementing virtual_node_name, call create_trees
            # with node 2 as the postreq and item 2 as the prereqs list. Return the original postreq because we are done
            if item_2.split('(')[0] == self.__AND_RELATIONSHIP or item_2.split('(')[0] == self.__OR_RELATIONSHIP or \
                    item_2.split('(')[0] == self.__SINGLE_RELATIONSHIP:
                nodes[1].set_name(random.randint(1, 101))
                nodes[1].set_virtual(nodes[1].VIRTUAL_TYPE)
                self.__create_trees(nodes[1], item_2)
                return postreq

        # If SINGLE relationship, create a node and set the name and grade to the item without an operator.
        # Return by calling create_trees with the node as postreq and its items as prereq list.
        if operator == self.__SINGLE_RELATIONSHIP:
            a = Node('', self.__SINGLE_RELATIONSHIP)
            removed_operator = prereqs.split('}')[0][1:]
            a.set_grade(removed_operator.split('#')[1])
            a.set_name(removed_operator.split('#')[0])
            postreq.add_prereq(a)
            return self.__create_trees(a, self.__find_items(a.get_name()))

    def __find_items(self, postreq_class_name):
        """
        Retrieves the prereqs list for a given postrequisite class name
        :param postreq_class_name: a string that contains the class name of a prereq
        :return: the cell from the prereqs column that matches the given postreq class name
        """
        for i, row in self.__READ_FILE.iterrows():
            if self.__READ_FILE.at[i, self.__POSTREQ] == postreq_class_name:
                return self.__READ_FILE.at[i, self.__PREREQ]
        return ''

    def __get_commas_for_split(self, operator, prereqs):
        """
        Retrieves the index of which comma should be used to separate the main logical expression for a prereq list
        :param operator: Which operator is being used in this prerequisite list (AND, OR, SINGLE)
        :param prereqs: The list of prereqs, not including the main operator or last parenthesis in the string
        :return: The comma position, in the prereqs string, where the two items split
        """
        open_paren = 0
        commas_needed = 1
        char_iter = iter(prereqs)
        for idx, char in enumerate(char_iter):
            if char == '(':
                open_paren += 1
                commas_needed += 1
            if char == ')':
                open_paren -= 1
            if idx > len(operator) and open_paren == 0:
                return commas_needed

    def process(self, postreq_class_name):
        """
        The public method for a TreeMaker object. Gets the prerequisite tree for a given postreq class name.
        :param postreq_class_name: A string containing the class name to retrieve the prerequisite tree for.
        :return: The headnode, whose name is the given postrequisite class name, for the prerequisite tree
        """
        class_node = Node(postreq_class_name, 'SINGLE')
        items = self.__find_items(postreq_class_name)
        self.__create_trees(class_node, items)
        return class_node

