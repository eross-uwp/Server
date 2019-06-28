import pandas as pd
from OneDepthCourseTreeGenerator import one_depth_tree
from anytree import RenderTree
forest = {}


if __name__ == '__main__':
    raw_data = pd.read_csv('..\\data\\Curriculum Structure.csv')
    class_list = list(raw_data.postreq.unique())  # all classes in postreq


