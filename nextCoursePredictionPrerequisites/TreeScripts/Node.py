class Node:
    __SINGLE_RELATIONSHIP = 'single'
    __AND_RELATIONSHIP = 'and'
    __OR_RELATIONSHIP = 'or'

    _prereq = []
    _coreq = []
    _postreq = []
    _name = ''
    _relationship = ''

    def __init__(self, name, relationship):
        if self.__check_relationship(relationship) == 1:
            self._relationship = relationship
        self.name = name

    def set_prereq(self, prereq):
        self._prereq = prereq

    def set_coreq(self, coreq):
        self._coreq = coreq

    def set_postreq(self, postreq):
        self._postreq = postreq

    def set_name(self, name):
        self._name = name

    def set_relationship(self, relationship):
        if self.__check_relationship(relationship) == 1:
            self._relationship = relationship

    def __check_relationship(self, relationship):
        if (relationship != self.__SINGLE_RELATIONSHIP and relationship != self.__AND_RELATIONSHIP and
                relationship != self.__OR_RELATIONSHIP):
            raise ValueError('An invalid Node relationship was passed. Must be \'single\', \'and\', or \'or\'')
        else:
            return 1

    def __copy__(self):
        return self
