"""
__Author__: Nate Braukhoff

__Purpose__: The Node will contain the following:
                Class Name,
                List of Nodes that will be it's Parents,
                Data Frame that will be have data from it's parents
"""
import pandas as pd
import numpy as np
import itertools

class Node:
    def __init__(self, name, children=None, grade=None, parents=None):
        """
        Create a Node with a name and an option of having a list of children and a grade. If the node has no children
        then _children will be set to none.
        :param name: string
        :param children: list of Nodes
        :param grade: Grade for the Node (Type: char)
        """
        if children is None:
            children = []

        if grade is None:
            grade = ''

        if parents is None:
            parents = []

        self._name = name
        self._children = children
        self._probability_table = pd.DataFrame()
        self._grade = grade
        self._parents = parents

    def get_name(self):
        """
        get_name will return the Node's attribute _name as a string
        :return: string
        """
        return str(self._name)

    def get_children(self):
        """
        get_children will return a list of all of the Nodes children
        :return: list of Nodes
        """
        return self._children

    def get_child(self, name_of_child):
        """
        get_child will iterate through the list of children and return the desired child. If a child is not found the
        method will return None. This method will also return None if _children is None
        :param name_of_child: The name of a child (Type: String)
        :return: Node or None
        """
        if len(self._children) != 0:
            for child in self._children:
                if name_of_child == child.get_name():
                    return child
        return None

    def get_grade(self):
        return self._grade

    def get_probability_table(self):
        return self._probability_table

    def set_grade(self, grade):
        self._grade = grade

    def get_parents(self):
        return self._parents

    def get_parent(self, name_of_parent):
        """
        get_parent searches through the list of parents and returns the parent the user is looking for. If the parent
        does not exist in the list then the method will return None.
        :param name_of_parent: String
        :return: parent of None
        """
        for parent in self._parents:
            if parent.get_name() == name_of_parent:
                return parent
        return None

    def add_child(self, child):
        """
        add_child will add a child to the end of _children. If _children is None then a new list will be created with
        the child in it.
        :param child: child node
        """
        if self._children is None:
            self._children = [child]
        else:
            self._children.append(child)

    def add_parent(self, parent):
        """
        add_parent will add the parent to the end of the parent list if the parents doesn't already exists in _parents.
        :param parent: Node
        """
        self._parents.append(parent)

    def remove_child(self, name_of_child):
        """
        remove_child will remove the node from the node's children's list
        :param name_of_child: String
        """
        for child in self._children:
            if child.get_name() == name_of_child:
                self._children.remove(child)
    '''
    def update_column(self, parent_name, column):
        """
        update_column will update the parents column for the Nodes DataFrame
        :param parent_name: string
        :param column: list
        """
        if self._probability_table.empty or parent_name in self._probability_table.columns:
            self._probability_table[parent_name] = column
        else:
            df = pd.DataFrame({parent_name: column})
            self._probability_table = pd.concat([self._probability_table, df], axis=1)
    '''
    def update_probability_table(self, data, scale):
        # Todo: Get Probability of each grad combination
        # Todo: return the probability for for the combination parent.

        return self._probability_table

    def get_all_combination(self, l):
        """
        get_all_combination takes in a list of items, then returns all possible combinations of each item.
        :param l: list of items
        :return: list of combinations
        """
        combination_list = []
        for i in list(itertools.product(l, repeat=len(self._parents))):
            combination_list.append(i)

        return combination_list

    def get_combination_probability(self, combination, data, predict):
        """
        get_combination_probability will return the probability of the node's state given it's parents' states.
        :param combination: list of strings
        :param data: DataFrame
        :param predict: state of Node
        :return:
        """
        df = data
        for i in range(0, len(self._parents)):
            df = df[df[self._parents[i].get_name()] == combination[i]]

        total = df.shape[0]
        occurrence = len(df[df[self._name] == predict])
        
        return occurrence / total

        

        # count how many time a grade shows up.
