# Using linear regression to predict if a student will graduate or not

'''
___authors___: Austin FitzGerald
'''

from sklearn import metrics
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import StratifyAndGenerateDatasets as sd

RESULTS_FOLDER = 'LinearRegressionResults\\'
GRAPH_FILE_PREFIX = 'LinearRegression_graph_'
RESULTS_TEXTFILE = 'LinearRegression_Results.txt'

first_term_X_train = []
second_term_X_train = []
third_term_X_train = []
first_term_X_test = []
second_term_X_test = []
third_term_X_test = []

first_term_y_train = []
second_term_y_train = []
third_term_y_train = []
first_term_y_test = []
second_term_y_test = []
third_term_y_test = []

x_train_array = [first_term_X_train, second_term_X_train, third_term_X_train]
x_test_array = [first_term_X_test, second_term_X_test, third_term_X_test]
y_train_array = [first_term_y_train, second_term_y_train, third_term_y_train]
y_test_array = [first_term_y_train, second_term_y_train, third_term_y_train]


def get_training_testing():
    for j in range(0, 3):
        for i in range(0, sd.NUMBER_FOLDS):
            x_train_array[j].append(pd.read_csv('data\\test_train\\' + sd.FILENAME_ARRAY[j] + sd.TRAIN_PREFIX + str(i+1) + '.csv')[sd.HEADERS_ARRAY[j]].values)
            y_train_array[j].append(pd.read_csv('data\\test_train\\' + sd.FILENAME_ARRAY[j] + sd.TRAIN_PREFIX + str(i+1) + '.csv')[[sd.GRADUATED_HEADER]].values)
            x_test_array[j].append(pd.read_csv('data\\test_train\\' + sd.FILENAME_ARRAY[j] + sd.TEST_PREFIX + str(i+1) + '.csv')[sd.HEADERS_ARRAY[j]].values)
            y_test_array[j].append(pd.read_csv('data\\test_train\\' + sd.FILENAME_ARRAY[j] + sd.TEST_PREFIX + str(i+1) + '.csv')[sd.GRADUATED_HEADER].values)


def lr_predict():
    np.random.seed(sd.RANDOM_SEED)
    model = LinearRegression()

    first_term_y_tests = []
    first_term_y_preds = []
    second_term_y_tests = []
    second_term_y_preds = []
    third_term_y_tests = []
    third_term_y_preds = []

    y_tests = [first_term_y_tests, second_term_y_tests, third_term_y_tests]
    y_preds = [first_term_y_preds, second_term_y_preds, third_term_y_preds]

    for j in range(0, 3):
        for i in range(0, sd.NUMBER_FOLDS):
            model.fit(x_train_array[j][i], y_train_array[j][i])
            y_pred = model.predict(x_test_array[j][i])
            y_tests[j] += list(y_test_array[j][i])
            y_preds[j] += list(y_pred)
        rr = metrics.r2_score(y_tests[j], y_preds[j])
        rmse = np.math.sqrt(metrics.mean_squared_error(y_tests[j], y_preds[j])) / 4
        with open(RESULTS_FOLDER + str(j) + RESULTS_TEXTFILE, "w") as text_file:
            text_file.write('R^2 = ' + str(rr) + ', RMSE = ' + str(rmse))

if __name__ == "__main__":
    get_training_testing()
    lr_predict()