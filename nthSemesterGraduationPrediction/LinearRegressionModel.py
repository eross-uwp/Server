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


#  Iterate through all possible training/testing files and store them in appropriate arrays.
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

    rr = []  # hold the R^2 and RMSE results for each term
    rmse = []  # |
    auc = []  # |

    y_tests = [[], [], []]  # hold the tests and predictions so we can graph them
    y_preds = [[], [], []]

    #  for each term, make a new model and fit it to all data in the folds. save the results and create graphs. for
    #  each term, calculate the R^2 and RMSE as well.
    for j in range(0, sd.NUM_TERMS):
        model = LinearRegression()
        for i in range(0, sd.NUMBER_FOLDS):
            model.fit(x_train_array[j][i], y_train_array[j][i])
            y_pred = model.predict(x_test_array[j][i])
            for idx, a in enumerate(y_pred):
                y_pred[idx] = sd.round_school(a)
            y_tests[j] += list(y_test_array[j][i])
            y_preds[j] += list(y_pred)
            plt.scatter((x_test_array[j][i])[:, 0], y_test_array[j][i], color='g', label='1st term')

            # TODO, not very extensible
            if j > 0:
                plt.scatter((x_test_array[j][i])[:, 2], y_test_array[j][i], color='r', label='2nd term')
            if j > 1:
                plt.scatter((x_test_array[j][i])[:, 4], y_test_array[j][i], color='b', label='3rd term')

            plt.plot((x_test_array[j][i])[:, 0], y_pred, color='k', label='predicted')
            plt.title('term #' + str(j + 1) + ', test #' + str(i + 1))
            plt.xlabel('GPA')
            plt.ylabel('graduation probability')
            plt.legend(loc='upper left')
            plt.savefig(RESULTS_FOLDER + GRAPH_FILE_PREFIX + str(j + 1) + '_' + str(i + 1))
            plt.close()

        rr.append(metrics.r2_score(y_tests[j], y_preds[j]))
        auc.append(metrics.roc_auc_score(y_tests[j], y_preds[j]))
        rmse.append(np.math.sqrt(metrics.mean_squared_error(y_tests[j], y_preds[j])))

    #  save all R^2 and RMSE results in one file with appropriate prefixes
    with open(RESULTS_FOLDER + RESULTS_TEXTFILE, "w") as text_file:
        for i in range(0, sd.NUM_TERMS):
            text_file.write(
                'term_' + str(i + 1) + ': R^2 = ' + str(rr[i]) + ', RMSE = ' + str(rmse[i]) + ', AUC = ' + str(
                    auc[i]) + '\n')


if __name__ == "__main__":
    get_training_testing()
    lr_predict()
