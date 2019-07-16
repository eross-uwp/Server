# Using gradient boosted trees to predict if a student will graduate or not

"""
___authors___: Austin FitzGerald
"""

from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import StratifyAndGenerateDatasets as sD

RESULTS_FOLDER = 'GBTClassifierResults\\'
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


def gbt_predict(term_number, criterion, learning_rate, loss, max_depth, max_features,
                min_samples_leaf, min_samples_split, n_estimators, subsample):
    np.random.seed(sD.RANDOM_SEED)

    y_tests = []  # hold combined tests and predictions for all folds
    y_preds = []
    y_grad_probs = []
    x_tests = []

    model = GradientBoostingClassifier(criterion=criterion, learning_rate=learning_rate, loss=loss, max_depth=max_depth,
                                       max_features=max_features,
                                       min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                       n_estimators=n_estimators, subsample=subsample, random_state=sD.RANDOM_SEED)
    for fold_num in range(0, sD.NUMBER_FOLDS):
        model.fit(x_train_array[term_number][fold_num], y_train_array[term_number][fold_num])
        y_pred = model.predict(x_test_array[term_number][fold_num])
        temp = model.predict_proba(x_test_array[term_number][fold_num])
        for t in temp:
            y_grad_probs.append(t[1])

        y_tests += list(y_test_array[term_number][fold_num])
        x_tests += list(x_test_array[term_number][fold_num])
        y_preds += list(y_pred)

    auc = metrics.roc_auc_score(y_tests, y_grad_probs)
    acc = metrics.accuracy_score(y_tests, y_preds)

    with open(RESULTS_FOLDER + RESULTS_TEXTFILE_PREFIX + str(term_number + 1) + '.txt', "w") as text_file:
        text_file.write(
            'AUC = ' + str(auc) + ', Accuracy = ' + str(acc))

    # save predictions (matching with tests) to files
    predictions = pd.DataFrame({'actual':y_tests, 'prediction': y_preds, 'prob of grad':y_grad_probs})
    predictions.to_csv(RESULTS_FOLDER + PREDICTION_OUTPUT_PREFIX + str(term_number + 1) + '.csv', index=False)


def get_all_term_stats():
    actuals = []
    predicted_grad_prob = []
    predicted_grad = []
    for i in range(0, sD.NUM_TERMS):
        term_predections_df = pd.read_csv(RESULTS_FOLDER+PREDICTION_OUTPUT_PREFIX+str(i+1) + '.csv')
        actuals += list(term_predections_df['actual'])
        predicted_grad_prob += list(term_predections_df['prob of grad'])
        predicted_grad += list(term_predections_df['prediction'])

    auc = metrics.roc_auc_score(actuals, predicted_grad_prob)
    acc = metrics.accuracy_score(actuals, predicted_grad)

    with open(RESULTS_FOLDER + RESULTS_TEXTFILE_PREFIX + '_all' + '.txt', "w") as text_file:
        text_file.write(
            'AUC = ' + str(auc) + ', Accuracy = ' + str(acc))


if __name__ == "__main__":
    get_training_testing()
    # TUNED
    gbt_predict(sD.FIRST_TERM, criterion='mae', learning_rate=0.1, loss='deviance', max_depth=3, max_features='log2',
                min_samples_leaf=0.2, min_samples_split=0.1, n_estimators=500, subsample=0.8)
    gbt_predict(sD.SECOND_TERM, criterion='friedman_mse', learning_rate=0.01, loss='deviance', max_depth=1,
                max_features=0.5,
                min_samples_leaf=0.1, min_samples_split=0.1, n_estimators=300, subsample=1.0)
    gbt_predict(sD.THIRD_TERM, criterion='mae', learning_rate=0.1, loss='deviance', max_depth=3, max_features=0.5,
                min_samples_leaf=0.1, min_samples_split=0.5, n_estimators=500, subsample=1.0)
    gbt_predict(sD.FOURTH_TERM, criterion='mae', learning_rate=0.1, loss='deviance', max_depth=1, max_features='log2',
                min_samples_leaf=0.1, min_samples_split=0.1, n_estimators=300, subsample=0.8)
    gbt_predict(sD.FIFTH_TERM, criterion='mae', learning_rate=0.5, loss='deviance', max_depth=3, max_features='log2',
                min_samples_leaf=0.1, min_samples_split=0.5, n_estimators=500, subsample=0.8)
    gbt_predict(sD.SIXTH_TERM, criterion='mae', learning_rate=0.1, loss='deviance', max_depth=3, max_features='log2',
                min_samples_leaf=0.1, min_samples_split=0.1, n_estimators=300, subsample=1.0)
    gbt_predict(sD.SEVENTH_TERM, criterion='friedman_mse', learning_rate=0.01, loss='deviance', max_depth=1,
                max_features=0.5,
                min_samples_leaf=0.1, min_samples_split=0.1, n_estimators=300, subsample=1.0)
    gbt_predict(sD.EIGHTH_TERM, criterion='friedman_mse', learning_rate=0.1, loss='deviance', max_depth=1,
                max_features=0.1,
                min_samples_leaf=0.1, min_samples_split=0.1, n_estimators=300, subsample=1.0)
    gbt_predict(sD.NINTH_TERM, criterion='friedman_mse', learning_rate=0.1, loss='deviance', max_depth=1,
                max_features='log2',
                min_samples_leaf=0.1, min_samples_split=0.1, n_estimators=300, subsample=1.0)
    gbt_predict(sD.TENTH_TERM, criterion='mae', learning_rate=0.1, loss='deviance', max_depth=1, max_features=0.1,
                min_samples_leaf=0.1, min_samples_split=0.1, n_estimators=300, subsample=0.8)

    get_all_term_stats()