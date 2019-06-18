"""
___authors___: Nate Braukhoff and Zhiwei Yang
"""

from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math
import urllib.request

NUM_TRAIN_TEST = 5  # 5 datasets
TRAIN_DATA_PATH = 'https://raw.githubusercontent.com/earos-uwp/Server/master/nextSemesterGpaPrediction/data/test_train/train_'  # Raw file address
TEST_DATA_PATH = 'https://raw.githubusercontent.com/earos-uwp/Server/master/nextSemesterGpaPrediction/data/test_train/test_'  # Raw file address

if __name__ == "__main__":
    for x in range(1, NUM_TRAIN_TEST + 1):
        url = (TRAIN_DATA_PATH + str(x) + '.csv')
        file = urllib.request.urlopen(url)
        train = pd.read_csv(file)  # Getting training dataset from Github

        url1 = (TEST_DATA_PATH + str(x) + '.csv')
        file1 = urllib.request.urlopen(url1)
        testData = pd.read_csv(file1)
        # Getting testing dataset from Github

        x = train.drop('current GPA', 1)
        y = np.array(train['current GPA'])

        model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, max_features=1, max_depth=4, loss='ls')
        model.fit(x, y)

        rmse_val = math.sqrt(mean_squared_error(testData['current GPA'].values, model.predict(testData.drop('current GPA', 1)))) / 4

        print('RMSE = ' + str(rmse_val))
