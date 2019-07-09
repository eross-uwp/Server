class Node:

    def __init__(self, name, relationship):
        if self.__check_relationship(relationship) == 1:
            self._relationship = relationship
        self._name = name
        self._prereq = []
        self._coreq = []
        self._postreq = []
        self.grade = ''

    def add_prereq(self, prereq):
        self._prereq.append(prereq)


    def add_coreq(self, coreq):
        self._coreq.append(coreq)

    def add_postreq(self, postreq):
        self._postreq.append(postreq)

    def set_name(self, name):
        self._name = name

    def set_grade(self, grade):
        self.grade = grade

    def get_grade(self):
        return self.grade

    def get_name(self):
        return self._name

    def set_relationship(self, relationship):
        if self.__check_relationship(relationship) == 1:
            self._relationship = relationship

    def __check_relationship(self, relationship):
        if (relationship != self.__SINGLE_RELATIONSHIP and relationship != self.__AND_RELATIONSHIP and
                relationship != self.__OR_RELATIONSHIP):
            raise ValueError('An invalid Node relationship was passed. Must be \'SINGLE\', \'AND\', or \'OR\'')
        else:
            return 1

    def __copy__(self):
        copy = Node(self._name, self._relationship)
        copy.add_prereq(self._prereq)
        copy.add_coreq(self._coreq)
        copy.add_postreq(self._postreq)
        return copy
