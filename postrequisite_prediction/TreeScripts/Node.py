"""
___authors___: Austin FitzGerald and Evan Majerus
"""


class Node:
    __SINGLE_RELATIONSHIP = 'SINGLE'
    __AND_RELATIONSHIP = 'AND'
    __OR_RELATIONSHIP = 'OR'
    VIRTUAL_TYPE = 1
    NON_VIRTUAL_TYPE = 0

    def __init__(self, name, relationship):
        if self.__check_relationship(relationship) == 1:
            self._relationship = relationship
        self._name = name
        self._prereqs = []
        self._coreq = []
        self._postreq = []
        self._grade = ''
        self._virtual = 0

    def add_prereq(self, prereq):
        self._prereqs.append(prereq)

    def add_coreq(self, coreq):
        self._coreq.append(coreq)

    def add_postreq(self, postreq):
        self._postreq.append(postreq)

    def set_name(self, name):
        self._name = name

    def set_virtual(self, virtual):
        if virtual != self.NON_VIRTUAL_TYPE and virtual != self.VIRTUAL_TYPE:
            raise ValueError('An invalid virtual parameter was passed. Must be 0 or 1')
        else:
            self._virtual = virtual

    def get_virtual(self):
        return self._virtual

    def set_grade(self, grade):
        self._grade = grade

    def get_grade(self):
        return self._grade

    def get_name(self):
        return self._name

    def does_have_prereq(self):
        if len(self._prereqs) > 0:
            return 1
        else:
            return 0

    def get_immediate_prereqs(self):
        temp_list = []
        for prereq in self._prereqs:
            if prereq.get_virtual() == self.VIRTUAL_TYPE:
                virtual_list = prereq.get_immediate_prereqs()
                for n in virtual_list:
                    temp_list.append(n)
            else:
                temp_list.append(prereq)

        return temp_list

    def get_all_prereqs(self):
        temp_list = []

        for prereq in self._prereqs:
            if prereq.get_virtual() == self.VIRTUAL_TYPE:
                virtual_list = prereq.get_immediate_prereqs()
                for n in virtual_list:
                    temp_list.append(n)
                    temp_prereq_list = n.get_all_prereqs()
                    for t in temp_prereq_list:
                        temp_list.append(t)
            else:
                temp_list.append(prereq)
                temp_prereq_list = prereq.get_all_prereqs()
                for t in temp_prereq_list:
                    temp_list.append(t)

        return temp_list

    def set_relationship(self, relationship):
        if self.__check_relationship(relationship) == 1:
            self._relationship = relationship

    def get_relationship(self):
        return self._relationship

    def __check_relationship(self, relationship):
        if (relationship != self.__SINGLE_RELATIONSHIP and relationship != self.__AND_RELATIONSHIP and
                relationship != self.__OR_RELATIONSHIP):
            raise ValueError('An invalid Node relationship was passed. Must be \'SINGLE\', \'AND\', or \'OR\'')
        else:
            return 1

    def __copy__(self):
        copy = Node(self._name, self._relationship)
        copy.add_prereq(self._prereqs)
        copy.add_coreq(self._coreq)
        copy.add_postreq(self._postreq)
        copy.set_grade(self._grade)
        copy.set_virtual(self._virtual)
        return copy