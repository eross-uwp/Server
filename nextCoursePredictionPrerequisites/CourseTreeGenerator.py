from treelib import Node, Tree
from anytree.importer import DictImporter
from anytree import RenderTree

if __name__ == '__main__':
    importer = DictImporter()
    data = {
        'a': 'root',
        'children': [{'a': 'x',
                      'children': [{'x': 'a'}]},
                     {'a': 'y',
                      'children': [{'y': 'x'}, {'y': 'z'}]},
                     {'a': 'z'}]
    }
    root = importer.import_(data)
    print(RenderTree(root))

