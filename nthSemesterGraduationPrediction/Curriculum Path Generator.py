import pandas as pd
import numpy as np
import urllib.request

TRAIN_DATA_PATH = ''
url = (TRAIN_DATA_PATH + '.csv')
file = urllib.request.urlopen(url)
trainData = pd.read_csv(file)  # Getting training dataset from Github