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
TRAIN_DATA_PATH = 'data\\/test_train\\train_'
# Raw file address
TEST_DATA_PATH = 'data\\test_train\\test_'

RANDOM_SEED = 313131


# Getting training dataset from Github
def get_train_data():
    return pd.read_csv(TRAIN_DATA_PATH + str(x) + '.csv')


# Getting testing dataset from Github
def get_test_data():
    return pd.read_csv(TEST_DATA_PATH + str(x) + '.csv')


if __name__ == "__main__":
    for x in range(1, NUM_TRAIN_TEST + 1):
        train = get_train_data()
        test = get_test_data()

        # set variables for the model
        x = train.drop('current GPA', 1)
        y = np.array(train['current GPA'])

        # create the model
        model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, max_features=1, max_depth=4, loss='ls',
                                          random_state=RANDOM_SEED)
        model.fit(x, y)

        # Calculate the Root mean squared
        rmse_val = math.sqrt(mean_squared_error(test['current GPA'].values,
                                                model.predict(test.drop('current GPA', 1)))) / 4

        print('RMSE = ' + str(rmse_val))
