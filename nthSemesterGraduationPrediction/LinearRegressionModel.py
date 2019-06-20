# Using linear regression to predict if a student will graduate or not

"""
___authors___: Austin FitzGerald
"""

from sklearn import metrics
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import StratifyAndGenerateDatasets as sd

RESULTS_FOLDER = 'LinearRegressionResults\\'
GRAPH_FILE_PREFIX = 'graph_term_'
RESULTS_TEXTFILE = 'LinearRegression_Results.txt'

x_train_array = [[], [], []]
x_test_array = [[], [], []]
y_train_array = [[], [], []]
y_test_array = [[], [], []]


def get_training_testing():
    for j in range(0, sd.NUM_TERMS):
        for i in range(0, sd.NUMBER_FOLDS):
            x_train_array[j].append(
                pd.read_csv('data\\test_train\\' + sd.FILENAME_ARRAY[j] + sd.TRAIN_PREFIX + str(i + 1) + '.csv')[
                    sd.HEADERS_ARRAY[j]].values)
            y_train_array[j].append(
                pd.read_csv('data\\test_train\\' + sd.FILENAME_ARRAY[j] + sd.TRAIN_PREFIX + str(i + 1) + '.csv')[
                    sd.GRADUATED_HEADER].values)
            x_test_array[j].append(
                pd.read_csv('data\\test_train\\' + sd.FILENAME_ARRAY[j] + sd.TEST_PREFIX + str(i + 1) + '.csv')[
                    sd.HEADERS_ARRAY[j]].values)
            y_test_array[j].append(
                pd.read_csv('data\\test_train\\' + sd.FILENAME_ARRAY[j] + sd.TEST_PREFIX + str(i + 1) + '.csv')[
                    sd.GRADUATED_HEADER].values)


def lr_predict():
    np.random.seed(sd.RANDOM_SEED)
    model = LinearRegression()

    rr = []
    rmse = []

    y_tests = [[], [], []]
    y_preds = [[], [], []]

    for j in range(0, sd.NUM_TERMS):
        for i in range(0, sd.NUMBER_FOLDS):
            model.fit(x_train_array[j][i], y_train_array[j][i])
            y_pred = model.predict(x_test_array[j][i])
            y_tests[j] += list(y_test_array[j][i])
            y_preds[j] += list(y_pred)
            plt.scatter((x_test_array[j][i])[:, 0], y_test_array[j][i], color='g', label='1st term')

            # TODO, not very extensible
            if j > 0:
                plt.scatter((x_test_array[j][i])[:, 2], y_test_array[j][i], color='r', label='2nd term')
            if j > 1:
                plt.scatter((x_test_array[j][i])[:, 4], y_test_array[j][i], color='b', label='3rd term')

            plt.plot((x_test_array[j][i])[:, 0], model.predict(x_test_array[j][i]), color='k', label='predicted')
            plt.title('term #' + str(j + 1) + ', test #' + str(i + 1))
            plt.xlabel('GPA')
            plt.ylabel('graduation probability')
            plt.legend(loc='upper left')
            plt.savefig(RESULTS_FOLDER + GRAPH_FILE_PREFIX + str(j + 1) + '_' + str(i + 1))
            plt.close()

        rr.append(metrics.r2_score(y_tests[j], y_preds[j]))
        rmse.append(np.math.sqrt(metrics.mean_squared_error(y_tests[j], y_preds[j])))

    with open(RESULTS_FOLDER + RESULTS_TEXTFILE, "w") as text_file:
        for i in range(0, sd.NUM_TERMS):
            text_file.write('term_' + str(i+1) + ': R^2 = ' + str(rr[i]) + ', RMSE = ' + str(rmse[i]) + '\n')


if __name__ == "__main__":
    get_training_testing()
    lr_predict()
