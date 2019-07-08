from anytree.importer import DictImporter
from anytree import AnyNode, RenderTree
import numpy as np
import pandas as pd


raw_data = pd.read_csv('..\\data\\Curriculum Structure.csv')
Children = 'children'

children_list_test = ['x', 'y', 'z']
SELF_KEY = 'key'
CORE = ''

forest = {}
new_forest = {}

def get_children_list(prereq_classes):
    temp_list=[]
    for each_element in prereq_classes:
        temp_list.append({SELF_KEY:each_element})
    return temp_list


def one_depth_tree(post_req, pre_reqs):
    CORE = str(post_req)
    tree_root = {SELF_KEY:CORE}
    tree_root['children'] = get_children_list(pre_reqs)
    importer = DictImporter()
    root = importer.import_(tree_root)
    return root

def all_prereq(post_course):


if __name__ == '__main__':
    class_list = list(raw_data.postreq.unique())  # all classes in postreq

    new_forest = forest
    for post_class in class_list:
        pre_classes = raw_data[raw_data.postreq == post_class][['prereq']]  # list of prereq for class
        forest[post_class] = one_depth_tree(post_class, pre_classes.values.tolist())

        #print(RenderTree(forester[post_class]))
    new_forest = forest
    for post_class in class_list:
        new_forest[post_class] = all_prereq(post_class)

        print(RenderTree(forest[post_class]))
        print('\n\n\n')
