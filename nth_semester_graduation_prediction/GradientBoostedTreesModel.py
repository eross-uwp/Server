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

    model = GradientBoostingClassifier(criterion=criterion, learning_rate=learning_rate, loss=loss, max_depth=max_depth,
                                       max_features=max_features,
                                       min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                       n_estimators=n_estimators, subsample=subsample, random_state=sD.RANDOM_SEED)
    for fold_num in range(0, sD.NUMBER_FOLDS):
        model.fit(x_train_array[term_number][fold_num], y_train_array[term_number][fold_num])
        y_pred = model.predict(x_test_array[term_number][fold_num])

        # round the graduation predictions, either 1 or 0
        for idx, a in enumerate(y_pred):
            y_pred[idx] = sD.round_school(a)

        y_tests += list(y_test_array[term_number][fold_num])

        y_preds += list(y_pred)
        plt.scatter((x_test_array[term_number][fold_num])[:, 0], y_test_array[term_number][fold_num], color='g',
                    label='1st term')

        # TODO, not very extensible
        if term_number > sD.FIRST_TERM:
            plt.scatter((x_test_array[term_number][fold_num])[:, 2], y_test_array[term_number][fold_num],
                        color='#e6194B',
                        label='2nd term')
        if term_number > sD.SECOND_TERM:
            plt.scatter((x_test_array[term_number][fold_num])[:, 4], y_test_array[term_number][fold_num],
                        color='#f58231',
                        label='3rd term')
        if term_number > sD.THIRD_TERM:
            plt.scatter((x_test_array[term_number][fold_num])[:, 6], y_test_array[term_number][fold_num],
                        color='#ffe119',
                        label='4th term')
        if term_number > sD.FOURTH_TERM:
            plt.scatter((x_test_array[term_number][fold_num])[:, 8], y_test_array[term_number][fold_num],
                        color='#bfef45',
                        label='5th term')
        if term_number > sD.FIFTH_TERM:
            plt.scatter((x_test_array[term_number][fold_num])[:, 10], y_test_array[term_number][fold_num],
                        color='#3cb44b',
                        label='6th term')
        if term_number > sD.SIXTH_TERM:
            plt.scatter((x_test_array[term_number][fold_num])[:, 12], y_test_array[term_number][fold_num],
                        color='#42d4f4',
                        label='7th term')
        if term_number > sD.SEVENTH_TERM:
            plt.scatter((x_test_array[term_number][fold_num])[:, 14], y_test_array[term_number][fold_num],
                        color='#4363d8',
                        label='8th term')
        if term_number > sD.EIGHTH_TERM:
            plt.scatter((x_test_array[term_number][fold_num])[:, 16], y_test_array[term_number][fold_num],
                        color='#911eb4',
                        label='9th term')
        if term_number > sD.NINTH_TERM:
            plt.scatter((x_test_array[term_number][fold_num])[:, 18], y_test_array[term_number][fold_num],
                        color='#f032e6',
                        label='10th term')

        plt.scatter((x_test_array[term_number][fold_num])[:, 0], y_pred, color='k', label='predicted')
        plt.title('term #' + str(term_number + 1) + ', test #' + str(fold_num + 1))
        plt.xlabel('GPA')
        plt.ylabel('graduation')
        plt.legend(loc='upper left')
        plt.savefig(RESULTS_FOLDER + GRAPH_FILE_PREFIX + str(term_number + 1) + '_' + str(fold_num + 1))
        plt.close()

    rr = metrics.r2_score(y_tests, y_preds)
    auc = metrics.roc_auc_score(y_tests, y_preds)
    acc = metrics.accuracy_score(y_tests, y_preds)
    rmse = np.math.sqrt(metrics.mean_squared_error(y_tests, y_preds))

    # save all statistical results with appropriate prefixes
    with open(RESULTS_FOLDER + RESULTS_TEXTFILE_PREFIX + str(term_number + 1) + '.txt', "w") as text_file:
        text_file.write(
            'R^2 = ' + str(rr) + ', RMSE = ' + str(rmse) + ', AUC = ' + str(auc) + ', Accuracy = ' + str(acc))

    # save predictions (matching with tests) to files
    predictions = pd.DataFrame({'graduation prediction': y_preds})
    predictions.to_csv(RESULTS_FOLDER + PREDICTION_OUTPUT_PREFIX + str(term_number + 1) + '.csv', index=False)


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
