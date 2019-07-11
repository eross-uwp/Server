from TreeScripts.TreeMaker import TreeMaker
import pandas as pd

__COMBINED_STRUCTURE = 'data\\combined_structure.csv'
__COMBINED_STRUCTURE_FILE = pd.read_csv(__COMBINED_STRUCTURE)

if __name__ == "__main__":
    treemaker = TreeMaker(__COMBINED_STRUCTURE)
