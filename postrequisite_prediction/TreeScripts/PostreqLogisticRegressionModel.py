import numpy as np
import pandas as pd

from TreeScripts.TreeMaker import TreeMaker
from TreeScripts.Node import Node

class PostreqLogisticRegressionModel:

    def __get_course_info(self, tree, grades):
        postrequisite = tree.get_name()
        prerequisite = tree.get_prereq()
        prerequisite_names = []
        pre_grades = []
        for j in prerequisite:
            prerequisite_names.append(prerequisite[j].get_name())
        for j, tier in grades.iterrows():
            if tier[postrequisite] != '':
                post_grade = tier[postrequisite]
            for k in prerequisite_names:
                pre_grades.append(tier[prerequisite_names[k]])
            if check_prerequisites(pre_grades):


    def __

if __name__ == "__main__":
    prerequisites = pd.read_csv('..\\data\\combined.csv')
    grades = pd.read_csv('..\\data\\student_grade_list.csv')
    prerequisite_tree_maker = TreeMaker(prerequisites)
    for i, row in prerequisites.iterrows():
        tree = prerequisite_tree_maker.process(row['postreq'])
        PostreqLogisticRegressionModel.get_course_info(tree, grades)