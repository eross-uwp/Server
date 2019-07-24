"""
__Author__: Nate Braukhoff and Evan Majerus

__Purpose__:
"""
import pandas as pd
import numpy as np
import itertools


class Node:
    def __init__(self, name, children=None, state=None, parents=None):
        """
        Create a Node with a name and an option of having a list of children and a grade. If the node has no children
        then _children will be set to none.
        :param name: string
        :param children: list of Nodes
        :param state: Grade for the Node (Type: char)
        """
        if children is None:
            children = []

        if state is None:
            state = ''

        if parents is None:
            parents = []

        self._name = name
        self._children = children
        self._cp_table = pd.DataFrame()
        self._state = state
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

    def get_state(self):
        return self._state

    def get_cp_table(self):
        return self._cp_table

    def set_grade(self, grade):
        self._state = grade

    def get_parents(self):
        return self._parents

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

    def get_combination_probability(self, filtered_data, predict):
        """
        get_combination_probability will return the probability of the node's state given it's parents' states.
        :param filtered_data: DataFrame
        :param predict: state of Node
        :return: float
        """
        number_of_rows = filtered_data.shape[0]
        if number_of_rows == 0:
            return 0
        occurrence = len(filtered_data[filtered_data[self._name] == predict])

        return occurrence / number_of_rows

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

    def update_cp_table(self, data, scale):
        """
        update_probability_table will up a DataFrame that will represent the Node's Conditional Probability Table
        (CP Table). This DataFrame will contain all of the probability of the Node's states given the parent's states.
        :param data: DataFrame
        :param scale: List of strings
        """
        cpt = self.create_data_frame_columns(scale)
        # When a Node has no parents then filtering based on parent states is not needed.
        if len(self._parents) == 0:
            row = []
            for i in range(0, len(scale)):
                row.append(self.get_combination_probability(data, scale[i]))
            cpt.loc[0] = row

            self._cp_table = cpt
        else:
            combo = self.get_all_combination(scale)

            for i in range(0, len(combo)):
                row = []
                row.extend(combo[i])
                for value in scale:
                    row.append(self.get_combination_probability(self.filter_data(data, combo[i]), value))
                cpt.loc[i] = row

            self._cp_table = cpt

    def create_data_frame_columns(self, scale):
        """
        create_data_frame_columns will add all the node's parent's names as columns in the DataFrame. After the parent's
        each value in the scale will also be a column after the parents name.
        :param scale: list of Strings
        :return: DataFrame
        """
        columns = []

        if len(self._parents) != 0:
            for parent in self._parents:
                columns.append(parent.get_name())

        columns.extend(scale)

        return pd.DataFrame(columns=columns)

    def filter_data(self, data, combination):
        """
        filter_data will filter the data frame based on the combination of the parents states.
        :param data: Data Frame. Pre condition: Node must have at least one parent
        :param combination: list of strings
        :return: Data Frame
        """
        df = data

        for i in range(0, len(self._parents)):
            df = df[df[self._parents[i].get_name()] == combination[i]]
        return df
