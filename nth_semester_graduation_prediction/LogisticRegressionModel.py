"""
___authors___: Austin FitzGerald
 Using logistic regression to predict if a student will graduate or not
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import StratifyAndGenerateDatasets as sD

RESULTS_FOLDER = 'LogisticRegressionResults\\'
GRAPH_FILE_PREFIX = 'graphs\\term_'
RESULTS_TEXTFILE_PREFIX = 'stats\\term_'
PREDICTION_OUTPUT_PREFIX = 'prediction output\\term_'

x_train_array = [[], [], [], [], [], [], [], [], [], []]
x_test_array = [[], [], [], [], [], [], [], [], [], []]
y_train_array = [[], [], [], [], [], [], [], [], [], []]
y_test_array = [[], [], [], [], [], [], [], [], [], []]


#  Iterate through all possible training/testing files and store them in appropriate arrays.
def get_training_testing():
    for j in range(0, sD.NUM_TERMS):
        for i in range(0, sD.NUMBER_FOLDS):
            x_train_array[j].append(
                pd.read_csv('data\\test_train\\' + sD.FILENAME_ARRAY[j] + sD.TRAIN_PREFIX + str(i + 1) + '.csv')[
                    sD.HEADERS_ARRAY[j]].values)
            y_train_array[j].append(
                pd.read_csv('data\\test_train\\' + sD.FILENAME_ARRAY[j] + sD.TRAIN_PREFIX + str(i + 1) + '.csv')[
                    sD.GRADUATED_HEADER].values)
            x_test_array[j].append(
                pd.read_csv('data\\test_train\\' + sD.FILENAME_ARRAY[j] + sD.TEST_PREFIX + str(i + 1) + '.csv')[
                    sD.HEADERS_ARRAY[j]].values)
            y_test_array[j].append(
                pd.read_csv('data\\test_train\\' + sD.FILENAME_ARRAY[j] + sD.TEST_PREFIX + str(i + 1) + '.csv')[
                    sD.GRADUATED_HEADER].values)


def lr_predict(term_number, c, penalty, solver):
    np.random.seed(sD.RANDOM_SEED)

    y_tests = []  # hold combined tests and predictions for all folds
    y_preds = []
    y_grad_probs = []
    x_tests = []

    model = LogisticRegression(random_state=sD.RANDOM_SEED, C=c, penalty=penalty, solver=solver)

    for fold_num in range(0, sD.NUMBER_FOLDS):
        model.fit(x_train_array[term_number][fold_num], y_train_array[term_number][fold_num])
        y_pred = model.predict(x_test_array[term_number][fold_num])
        temp = model.predict_proba(x_test_array[term_number][fold_num])
        for t in temp:
            y_grad_probs.append(t[1])

        y_tests += list(y_test_array[term_number][fold_num])
        x_tests += list(x_test_array[term_number][fold_num])
        y_preds += list(y_pred)

    # save predictions (matching with tests) to files
    predictions = pd.DataFrame({'actual':y_tests, 'prediction': y_preds, 'prob of grad':y_grad_probs})
    predictions.to_csv(RESULTS_FOLDER + PREDICTION_OUTPUT_PREFIX + str(term_number + 1) + '.csv', index=False)


if __name__ == "__main__":
    get_training_testing()
    lr_predict(sD.FIRST_TERM, c=11.288378916846883, penalty='l1', solver='liblinear')
    lr_predict(sD.SECOND_TERM, c=0.615848211066026, penalty='l2', solver='liblinear')
    lr_predict(sD.THIRD_TERM, c=1.0, penalty='l1', solver='liblinear')
    lr_predict(sD.FOURTH_TERM, c=1.0, penalty='l1', solver='liblinear')
    lr_predict(sD.FIFTH_TERM, c=1.0, penalty='l1', solver='liblinear')
    lr_predict(sD.SIXTH_TERM, c=1.0, penalty='l1', solver='liblinear')
    lr_predict(sD.SEVENTH_TERM, c=1.0, penalty='l1', solver='liblinear')
    lr_predict(sD.EIGHTH_TERM, c=1.0, penalty='l1', solver='liblinear')
    lr_predict(sD.NINTH_TERM, c=1.0, penalty='l1', solver='liblinear')
    lr_predict(sD.TENTH_TERM, c=1.0, penalty='l1', solver='liblinear')
