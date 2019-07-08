from anytree.importer import DictImporter
from anytree import RenderTree
import numpy as np
import pandas as pd


raw_data = pd.read_csv('..\\data\\Curriculum Structure.csv')
Children = 'children'

children_list_test = ['x', 'y', 'z']
SELF_KEY = 'key'
core = ''

forester = {}


def get_children_list(prereq_class):
    temp_list=[]
    for each_element in prereq_class:
        temp_list.append({SELF_KEY:each_element})
    return temp_list


def one_depth_tree(post_req, pre_reqs):
    core = str(post_req)
    tree_root = {SELF_KEY:core}
    tree_root['children'] = get_children_list(pre_reqs)
    importer = DictImporter()
    root = importer.import_(tree_root)
    return root


if __name__ == '__main__':
    class_list = list(raw_data.postreq.unique())  # all classes in postreq

    for postClass in class_list:
        pre_classes = raw_data[raw_data.postreq == postClass][['prereq']]  # list of prereq for class
        forester[postClass] = one_depth_tree(postClass, pre_classes.values.tolist())

        print(RenderTree(forester[postClass]))

