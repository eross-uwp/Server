"""
___authors___: Zhiwei Yang
"""

import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

PREV_GPA = 'prev GPA'
CURR_GPA = 'current GPA'
TRAIN_DATA_PATH = 'data\\test_train\\train_'  # Raw file address
TEST_DATA_PATH = 'data\\test_train\\test_'  # Raw file address
NUM_TRAIN_TEST = 5  # 5 datasets
RESULTS_FOLDER = 'ZeroRModelResults\\'
RESULTS_TEXTFILE = 'ZeroR_Results.txt'


def predict(original_data):
    # Finds the mean of original_data
    sum_of_gpa = 0
    for row in original_data:
        sum_of_gpa = sum_of_gpa + row
    prediction = sum_of_gpa/original_data.size
    return prediction


if __name__ == "__main__":
    for x in range(1, NUM_TRAIN_TEST + 1):
        trainData = pd.read_csv(TRAIN_DATA_PATH + str(x) + '.csv')  # Getting training dataset from Github

        predictGPA = predict(trainData[PREV_GPA])
        # Run the ZeroRModel predict function

        testData = pd.read_csv(TEST_DATA_PATH + str(x) + '.csv')
        # Getting training dataset from Github

        rmse_val = math.sqrt(mean_squared_error(testData[CURR_GPA].values.reshape(1, testData[CURR_GPA].size),
                                                np.full((1, testData[CURR_GPA].size),
                                                        predictGPA))) / 4  # normalized by 4.0 gpa scale
        # Saving the RMSE to a text file.
        with open(RESULTS_FOLDER + RESULTS_TEXTFILE, "w") as text_file:
            text_file.write('RMSE = ' + str(rmse_val))
