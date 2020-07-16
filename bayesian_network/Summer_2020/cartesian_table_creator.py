"""
__Author__: Nick Tiede

__Purpose__: To create cartesian product tables for many uses
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from itertools import product

import pandas as pd


# Creates the cartesian product of ints given by the num_values, converts the list of tuples to a list of lists,
# then to a DataFrame, if formatted: then converts all of the data to ints to get rid of decimals, then to strings
def create_cartesian_table(num_values, num_cols, formatted=True):
    if formatted:
        return pd.DataFrame(map(list, product(range(0, num_values), repeat=num_cols))).applymap(int).applymap(str)
    else:
        return pd.DataFrame(map(list, product(range(0, num_values), repeat=num_cols)))
