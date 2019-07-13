"""
__Author__: Nate Braukhoff

__Purpose__: The Node will contain the following:
                Class Name,
                List of Nodes that will be it's Parents,
                Data Frame that will be have data from it's parents
"""
import pandas as pd
import numpy as np


class Node:
    def __init__(self, name, children=None, grade=None):
        """
        Create a Node with a name and an option of having a list of children and a grade. If the node has no children
        then _children will be set to none.
        :param name: string
        :param children: list of Nodes
        :param grade: Grade for the Node (Type: char)
        """
        self._name = name
        self._children = children
        self._probability_table = pd.DataFrame()
        self._grade = grade

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
        if self._children is not None:
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

    def get_serial_probability(self, parent, grand_parent):
        # Todo: Get All Combinations of Grades
        # Todo: Get Probability of each grad combination
        # Todo: return the probability for for the combination parent and grand_parent

        return self._probability_table
