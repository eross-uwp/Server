"""
__Author__: Nick Tiede

__Purpose__: Read and write from csv files formatted for the bayesian network and pomegranate
"""

import pandas as pd


# Reads training or testing data from a csv and returns a pandas DataFrame
def read_data_csv(file):
    try:
        return pd.read_csv(file, na_filter=False).replace('', -1).applymap(int).applymap(str).replace('-1', 'nan')

    except:
        print('Could not open/read file: ' + file)


# Saves information to csv with specified file name as a pandas DataFrame
def write_data(data, file):
    pd.DataFrame(data).to_csv(file)
