import numpy as np
import pandas as pd

from TreeScripts.TreeMaker import TreeMaker
from TreeScripts.Node import Node


class PostreqLogisticRegressionModel:
    __STUDENT_ID = 'student_id'
    __CUMULATIVE_GPA = 'cumulative_gpa'
    __PREV_TERM_GPA = 'prev_term_gpa'
    __STRUGGLE = 'struggle'
    __EXTREME_STRUGGLE = 'extreme_struggle'
    __TERM_DIFFERENCE = 'term_difference'

    def __create_data_frame(self , tree, grades):
        postrequisite = tree.get_name()
        data_frame = pd.DataFrame(columns=[self.__STUDENT_ID, postrequisite])
        prerequisite = tree.get_prereq()
        for j in prerequisite:
            data_frame[prerequisite[j].get_name()] = ''
        data_frame[self.__CUMULATIVE_GPA] = ''
        data_frame[self.__PREV_TERM_GPA] = ''
        data_frame[self.__STRUGGLE] = ''
        data_frame[self.__EXTREME_STRUGGLE] = ''
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
                
                data_frame_row = data_frame_row + 1
        return data_frame

    def __taken_prereq(self, index, grades, prerequisite):
        for k in prerequisite:
            if grades.at[index, prerequisite[k]] != '':
                return True
        return False


if __name__ == "__main__":
    structure = pd.read_csv('..\\data\\combined_structure.csv')
    grades = pd.read_csv('..\\data\\student_grade_list.csv')
    prerequisite_tree_maker = TreeMaker(structure)
    for i, row in structure.iterrows():
        tree = prerequisite_tree_maker.process(row['postreq'])
        data_frame = PostreqLogisticRegressionModel.__create_data_frame(tree, grades)
