dict1={'s':'s', 's':'s'}
#dict1['jack']='444444'
print(dict1)

from anytree.importer import DictImporter
from anytree import RenderTree

SELF_KEY = 'kkk'
children = ['x', 'y', 'z']
grand_children_X = ['d', 'e']
grand_children_Y = ['f', 'g']
layers = [grand_children_X, grand_children_Y]
CORE = 'abc'

tree_root = {SELF_KEY:CORE}
def get_children_list():
    temp_list=[]
    for each_element in children:
        temp_list.append({SELF_KEY:each_element})
    return temp_list

if __name__ == '__main__':
    for each_lay
    tree_root['children'] = get_children_list()
    importer = DictImporter()
    root = importer.import_(tree_root)
    print(RenderTree(root))