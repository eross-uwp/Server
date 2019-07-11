import pandas as pd

from TreeScripts.TreeMaker import TreeMaker
from TreeScripts.Node import Node


class PostreqLinearRegressionModel:
    __STUDENT_ID = 'student_id'
    __CUMULATIVE_GPA = 'cumulative_gpa'
    __PREV_TERM_GPA = 'prev_term_gpa'
    __STRUGGLE = 'struggle'
    __TERM_DIFFERENCE = 'term_difference'

    def __create_data_frame(self, tree, grades):
        postrequisite = tree.get_name()
        data_frame = pd.DataFrame(columns=[self.__STUDENT_ID, postrequisite])
        prerequisite = tree.get_prereq()
        for j in prerequisite:
            data_frame[prerequisite[j].get_name()] = ''
        data_frame[self.__CUMULATIVE_GPA] = ''
        data_frame[self.__PREV_TERM_GPA] = ''
        data_frame[self.__STRUGGLE] = ''
        data_frame[self.__TERM_DIFFERENCE] = ''
        data_frame = self.__get_student_info(data_frame, grades, postrequisite, prerequisite)
        return data_frame

    def __get_student_info(self, data_frame, grades, postrequisite, prerequisite):
        data_frame_row = 1
        for j, tier in grades.iterrows():
            if grades.at[j, postrequisite] != '' and self.__taken_prereq(j, grades, prerequisite):
                data_frame.at[data_frame_row, self.__STUDENT_ID] = grades.at[j, self.__STUDENT_ID].split(',')[1]
                postreq_term = grades.at[j, self.__STUDENT_ID].split(',')[0]
                earliest_term = 2000
                for k in prerequisite:
                    if grades.at[j, prerequisite[k]] != '':
                        data_frame.at[data_frame_row, prerequisite[k]] = grades.at[j, prerequisite[k]].split(',')[1]
                        if grades.at[j, self.__STUDENT_ID].split(',')[0] < earliest_term:
                            earliest_term = grades.at[j, self.__STUDENT_ID].split(',')[0]
                data_frame.at[data_frame_row, self.__TERM_DIFFERENCE] = postreq_term - earliest_term
                data_frame = self.get_cumulative_gpa(data_frame, j, earliest_term)
                data_frame = self.get_prev_term_gpa(data_frame, j, earliest_term)
                data_frame.at[j, self.__STRUGGLE] = self.__have_struggled(j, earliest_term)
                data_frame_row = data_frame_row + 1
        return data_frame

    def __taken_prereq(self, index, grades, prerequisite):
        for k in prerequisite:
            if grades.at[index, prerequisite[k]] != '':
                return True
        return False

    def get_cumulative_gpa(self, data_frame, index, term):
        cumulative = pd.read_csv('..\\data\\cumulative_gpa.csv').fillna('')
        term_subtractor = 1
        while cumulative[index, term - term_subtractor] == '':
            term_subtractor = term_subtractor + 1
        data_frame.at[index, self.__CUMULATIVE_GPA] = cumulative[index, term - term_subtractor]
        return data_frame

    def get_prev_term_gpa(self, data_frame, index, term):
        prev_term_gpa = pd.read_csv('..\\data\\term_gpa.csv').fillna('')
        term_subtractor = 1
        while prev_term_gpa[index, term - term_subtractor] == '':
            term_subtractor = term_subtractor + 1
        data_frame.at[index, self.__PREV_TERM_GPA] = prev_term_gpa[index, term - term_subtractor]
        return data_frame

    def __have_struggled(self, index, term):
        struggle = pd.read_csv('..\\data\\strugling_per_term.csv').fillna('')
        prev_terms = 1
        value = ''
        while struggle.at[0, prev_terms] != struggle.at[0, term]:
            if struggle.at[index, prev_terms] == 'E':
                value = 'E'
            if struggle.at[index, prev_terms] == 'S' and value != 'E':
                value = 'S'
            if struggle.at[index, prev_terms] == 'G' and (value != 'E' or value != 'S'):
                value = 'G'
            prev_terms = prev_terms + 1
        return value


if __name__ == "__main__":
    structure = pd.read_csv('..\\data\\combined_structure.csv').fillna('')
    grades = pd.read_csv('..\\data\\student_grade_list_with_terms.csv').fillna('')
    prerequisite_tree_maker = TreeMaker('..\\data\\combined_structure.csv')
    for i, row in structure.iterrows():
        tree = prerequisite_tree_maker.process(row['postreq'])
        data_frame = PostreqLinearRegressionModel.__create_data_frame(tree, grades)
        data_frame.to_csv('..\\data\\LinearRegressionCSV\\'
                          + "".join([c for c in tree.get_name() if c.isalpha() or c.isdigit() or c == ' ']).rstrip()
                          + '.csv')
    print("Done!")
