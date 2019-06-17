# ----------------------------------------------------------------------------------------------------------------------
# ___authors___: Zhiwei Yang
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from nextSemesterGpaPrediction.data.ZeroRModel.ZeroRModel import predict
import urllib.request
from sklearn.metrics import mean_squared_error
import math


PREV_GPA = 'prev GPA'
CURR_GPA = 'current GPA'
TRAIN_DATA_PATH = 'https://raw.githubusercontent.com/earos-uwp/Server/master/nextSemesterGpaPrediction/data/' \
                  'test_train/train_'           # Raw file address
TEST_DATA_PATH = 'https://raw.githubusercontent.com/earos-uwp/Server/master/nextSemesterGpaPrediction/data/' \
                 'test_train/test_'             # Raw file address
NUM_TRAIN_TEST = 5                              # 5 datasets


if __name__ == "__main__":
    for x in range(1, NUM_TRAIN_TEST + 1):
        url = (TRAIN_DATA_PATH + str(x) + '.csv')
        file = urllib.request.urlopen(url)
        trainData = pd.read_csv(file)           # Getting training dataset from Github

        predictGPA = predict(trainData[PREV_GPA])
                                                # Run the ZeroRModel predict function

        testData = pd.read_csv(TEST_DATA_PATH + str(x) + '.csv')
                                                # Getting training dataset from Github

        rmse_val = math.sqrt(mean_squared_error(testData[CURR_GPA].values.reshape(1, testData[CURR_GPA].size),
                                      np.full((1, testData[CURR_GPA].size), predictGPA)))
        print('test ' + str(x) + ' rsme is ' + str(rmse_val))
                                                # Calculating the RSME