from treelib import Node, Tree
import numpy as np
import pandas as pd

DATA_FOLDER = '..\\data\\Curriculum Structure.csv'

curriculum_structure = pd.read_csv(DATA_FOLDER)
forest = np.zeros(0)
for postreq in curriculum_structure.iterrows():
    curriculum_structure.at[postreq]['postreq']
    for course in postreq.values:



