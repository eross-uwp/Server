import random
import sys
import pandas as pd

from TreeScripts.Node import Node

class FileReader:
    __SINGLE_RELATIONSHIP = 'SINGLE'
    __AND_RELATIONSHIP = 'AND'
    __OR_RELATIONSHIP = 'OR'
    __POSTREQ = 'postreq'
    __PREREQ = 'prereqs'
    __RELATIONSHIP = pd.read_csv('..\\data\\se.csv')

    def create_trees(self, postreq, items):
        # if and/or
        if items.split('(')[0] == self.__AND_RELATIONSHIP or items.split('(')[0] == self.__OR_RELATIONSHIP:
            operator = items.split('(')[0]
            nodes = [Node('', items.split('(')[0]), Node('', items.split('(')[0])]
            postreq.add_prereq(nodes[0])
            postreq.add_prereq(nodes[1])
            for idx,item in enumerate(items.split(',')):
                removed_operator = item[(len(operator)+1):]
                # if item is a class
                if removed_operator[0] == '{':
                    # get class name
                    nodes[idx].set_name(removed_operator[removed_operator.find('{')+1 : removed_operator.find('}')])
                    return self.create_trees(nodes[idx], self.find_items(nodes[idx].get_name()))
                if item.split('(')[0] == self.__AND_RELATIONSHIP or item.split('(')[0] == self.__OR_RELATIONSHIP or item.split('(')[0] == self.__SINGLE_RELATIONSHIP:
                    nodes[idx].set_name(random.randint(1,101))
                    return self.create_trees(nodes[idx], item)
                if item == '':
                    return postreq
        if items.split('(')[0] == self.__SINGLE_RELATIONSHIP:
            a = Node('', self.__SINGLE_RELATIONSHIP)
            removed_operator = a.get_name()[(len(self.__SINGLE_RELATIONSHIP) + 1):]
            a.set_name(removed_operator[removed_operator.find('{') + 1: removed_operator.find('}')])
            postreq.add_prereq(a)
            return self.create_trees(a, self.find_items(a.get_name()))

    def find_items(self, class_name):
        for i, rows in self.__RELATIONSHIP.iterrows():
            if self.__RELATIONSHIP.at[i, self.__POSTREQ] == class_name:
                return self.__RELATIONSHIP.at[i, self.__PREREQ]
        return ''

    def do_stuff(self):
        head_nodes = []
        for i, row in self.__RELATIONSHIP.iterrows():
            items = self.find_items(self.__RELATIONSHIP.at[i, self.__POSTREQ])
            test = items.split('(')[0]
            head_nodes.append(Node(self.__RELATIONSHIP.at[i, self.__POSTREQ], test))
            self.create_trees(head_nodes[i], items)


if __name__ == "__main__":
    sys.setrecursionlimit(1908)
    aay = FileReader()
    aay.do_stuff()
    print("DONE!")
