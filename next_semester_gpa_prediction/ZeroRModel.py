"""
___authors___: Zhiwei Yang
"""

import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

PREV_GPA = 'prev GPA'
CURR_GPA = 'next GPA'
TRAIN_DATA_PATH = 'data\\test_train\\train_'  # Raw file address
TEST_DATA_PATH = 'data\\test_train\\test_'  # Raw file address
NUM_TRAIN_TEST = 5  # 5 datasets
RESULTS_FOLDER = 'ZeroRModelResults\\'
PREDICTION_OUTPUT_PREFIX = 'ZeroR results'
RESULTS_TEXTFILE = 'ZeroR_Results.txt'


def predict(original_data):
    # Finds the mean of original_data
    sum_of_gpa = 0
    for row in original_data:
        sum_of_gpa = sum_of_gpa + row
    prediction = sum_of_gpa/original_data.size
    return prediction

flatten = lambda l: [item for sublist in l for item in sublist]

if __name__ == "__main__":
    prediction_array = np.zeros(0)
    test_prediction = pd.DataFrame()
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

        temp_predict =  np.full((1, testData[CURR_GPA].size), predictGPA)

        if x == 1:
            prediction_array = temp_predict
        else:
            prediction_array = np.concatenate((prediction_array, temp_predict), axis=1)

        with open(RESULTS_FOLDER + RESULTS_TEXTFILE, "w") as text_file:
            text_file.write('RMSE = ' + str(rmse_val))

        predictions = pd.DataFrame({'prediction': flatten(prediction_array)})
        if x == NUM_TRAIN_TEST:
            for y in range(1, NUM_TRAIN_TEST + 1):
                testData = pd.read_csv(TEST_DATA_PATH + str(x) + '.csv')
                if y == 1:
                    partial_test_prediction = testData
                else:
                    partial_test_prediction = pd.concat([partial_test_prediction, testData], axis=0)
                if y==NUM_TRAIN_TEST:
                    partial_test_prediction.to_csv(RESULTS_FOLDER + 'totalt_test' + '.csv', index=False)
            predictions = pd.DataFrame({'prediction': flatten(prediction_array)})
            predictions.to_csv(RESULTS_FOLDER + PREDICTION_OUTPUT_PREFIX + '.csv', index=False)
        # test_prediction = pd.concat([])

    # test_prediction.to_csv(RESULTS_FOLDER + PREDICTION_OUTPUT_PREFIX + '.csv', index=False)
