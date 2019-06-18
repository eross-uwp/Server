"""
___authors___: Nate Braukhoff and Zhiwei Yang
"""

from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math
import urllib.request

NUM_TRAIN_TEST = 5  # 5 data sets
# Raw file address
TRAIN_DATA_PATH = 'https://raw.githubusercontent.com/earos-uwp/Server/master/nextSemesterGpaPrediction/data' \
                  '/test_train/train_'
# Raw file address
TEST_DATA_PATH = 'https://raw.githubusercontent.com/earos-uwp/Server/master/nextSemesterGpaPrediction/data' \
                 '/test_train/test_'


# Getting training dataset from Github
def get_train_data():
    url = (TRAIN_DATA_PATH + str(x) + '.csv')
    file = urllib.request.urlopen(url)

    return pd.read_csv(file)


# Getting testing dataset from Github
def get_test_data():
    url1 = (TEST_DATA_PATH + str(x) + '.csv')
    file1 = urllib.request.urlopen(url1)

    return pd.read_csv(file1)


if __name__ == "__main__":
    for x in range(1, NUM_TRAIN_TEST + 1):
        train = get_train_data()
        test = get_test_data()

        # set variables for the model
        x = train.drop('current GPA', 1)
        y = np.array(train['current GPA'])

        # create the model
        model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, max_features=1, max_depth=4, loss='ls')
        model.fit(x, y)

        # Calculate the Root mean squared
        rmse_val = math.sqrt(mean_squared_error(test['current GPA'].values,
                                                model.predict(test.drop('current GPA', 1)))) / 4

        print('RMSE = ' + str(rmse_val))
