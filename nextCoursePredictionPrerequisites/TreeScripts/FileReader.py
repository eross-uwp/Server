import pandas as pd

from TreeScripts.Node import Node


class FileReader:
    __SINGLE_RELATIONSHIP = 'SINGLE'
    __AND_RELATIONSHIP = 'AND'
    __OR_RELATIONSHIP = 'OR'
    __POSTREQ = 'prereqs'
    __PREREQ = 'postreq'
    __RELATIONSHIP = pd.read('..\\data\\new.csv')

    def get_prerequisites(self):


    def read_string(self, prerequisite):
        if prerequisite[2] == self.__OR_RELATIONSHIP:
            prerequisite.lstrip(self.__OR_RELATIONSHIP)
            prerequisite.rstrip(')')
            prerequisite.lstrip('(')
            prerequisite.split(',')
            self.read_string(prerequisite[0])
            self.read_string(prerequisite[2])

        elif prerequisite[3] == self.__AND_RELATIONSHIP:
            prerequisite.lstrip(self.__AND_RELATIONSHIP)
            prerequisite.rstrip(')')
            prerequisite.lstrip('(')
            prerequisite.split(',')
            self.read_string(prerequisite[0])
            self.read_string(prerequisite[2])

        elif prerequisite[6] == self.__SINGLE_RELATIONSHIP:
            prerequisite.lstrip(self.__SINGLE_RELATIONSHIP)
            prerequisite.rstrip(')')
            prerequisite.lstrip('(')
            prerequisite.split(',')
            self.read_string(prerequisite[0])
        else:
            return prerequisite

    def find_relationship(self, prerequisite):
        if prerequisite[2] == self.__OR_RELATIONSHIP:
            return self.__OR_RELATIONSHIP
        elif prerequisite[3] == self.__AND_RELATIONSHIP:
            return self.__AND_RELATIONSHIP
        elif prerequisite[6] == self.__SINGLE_RELATIONSHIP:
            return self.__SINGLE_RELATIONSHIP

    if __name__ == "__main__":
        head_nodes = []
        for i, row in __RELATIONSHIP.iterrows():
            head_nodes[i] = Node(__RELATIONSHIP.at[i, __POSTREQ],
                                 find_relationship(__RELATIONSHIP.at[i, __PREREQ]))
            head_nodes[i].add_prereq(get_prerequisites(__RELATIONSHIP[i, __PREREQ]))
