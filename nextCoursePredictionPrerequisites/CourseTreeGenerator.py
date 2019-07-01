import pandas as pd
from anytree.importer import DictImporter

raw_data = pd.read_csv('data\\Curriculum Structure.csv')
Children = 'children'

children_list_test = ['x', 'y', 'z']
SELF_KEY = 'key'
CORE = ''

forester = {}
def get_children_list(prereq_class):
    temp_list=[]
    for each_element in prereq_class:
        temp_list.append({SELF_KEY:each_element})
    return temp_list


if __name__ == '__main__':
    class_list = list(raw_data.postreq.unique())  # all classes in postreq
    for postClass in class_list:
        post_reqs = raw_data[raw_data.postreq == postClass][['prereq']]  # list of prereq for class
        CORE = str(postClass)
        tree_root = {SELF_KEY:CORE}
        tree_root['children'] = get_children_list(post_reqs.values.tolist())
        importer = DictImporter()
        root = importer.import_(tree_root)
        #print(RenderTree(root))
        forester[CORE] = root
        #print('\n\n\n')
    print(forester)




