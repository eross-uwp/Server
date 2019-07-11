import numpy as np
import pandas as pd

from TreeScripts.TreeMaker import TreeMaker
from TreeScripts.Node import Node


class PostreqLogisticRegressionModel:
    def __create_data_frame(self , tree, grades):
        postrequisite = tree.get_name()
        data_frame = pd.DataFrame(columns=['student_id', postrequisite])
        prerequisite = tree.get_prereq()
        for j in prerequisite:
            data_frame[prerequisite[j].get_name]
        data_frame['struggle', 'extreme_struggle', 'term_difference']
        return data_frame


if __name__ == "__main__":
    prerequisites = pd.read_csv('..\\data\\combined_structure.csv')
    grades = pd.read_csv('..\\data\\student_grade_list.csv')
    prerequisite_tree_maker = TreeMaker(prerequisites)
    for i, row in prerequisites.iterrows():
        # data_frame = pd.DataFrame(columns = ['student_id', 'pre_one', 'pre_one_grade', 'pre_two', 'pre_two_grade',
                                            # 'pre_three', 'pre_three_grade', 'pre_four', 'pre_four_grade',
                                            # 'cumulative_gpa', 'prev_term_gpa', 'struggle', 'extreme_struggle',
                                            # 'postreq', 'postreg_grade', 'term_difference'])
        tree = prerequisite_tree_maker.process(row['postreq'])
        data_frame = PostreqLogisticRegressionModel.__create_data_frame(tree, grades)
        PostreqLogisticRegressionModel.__get_student_info(data_frame, grades)