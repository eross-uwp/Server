# Using gradient boosted trees to predict if a student will graduate or not

"""
___authors___: Austin FitzGerald
"""

from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import StratifyAndGenerateDatasets as sd

RESULTS_FOLDER = 'GBTClassifierResults\\'
GRAPH_FILE_PREFIX = 'graphs\\term_'
RESULTS_TEXTFILE_PREFIX = 'stats\\term_'
PREDICTION_OUTPUT_PREFIX = 'prediction output\\term_'

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


def gbt_predict(term_number, criterion, learning_rate, loss, max_depth, max_features,
                min_samples_leaf, min_samples_split, n_estimators, subsample):
    np.random.seed(sd.RANDOM_SEED)

    y_tests = []  # hold combined tests and predictions for all folds
    y_preds = []

    model = GradientBoostingClassifier(criterion=criterion, learning_rate=learning_rate, loss=loss, max_depth=max_depth,
                                       max_features=max_features,
                                       min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                       n_estimators=n_estimators, subsample=subsample, random_state=sd.RANDOM_SEED)
    for fold_num in range(0, sd.NUMBER_FOLDS):
        model.fit(x_train_array[term_number][fold_num], y_train_array[term_number][fold_num])
        y_pred = model.predict(x_test_array[term_number][fold_num])

        # round the graduation predictions, either 1 or 0
        for idx, a in enumerate(y_pred):
            y_pred[idx] = sd.round_school(a)

        y_tests += list(y_test_array[term_number][fold_num])

        y_preds += list(y_pred)
        plt.scatter((x_test_array[term_number][fold_num])[:, 0], y_test_array[term_number][fold_num], color='g',
                    label='1st term')

        # TODO, not very extensible
        if term_number > sd.FIRST_TERM:
            plt.scatter((x_test_array[term_number][fold_num])[:, 2], y_test_array[term_number][fold_num], color='r',
                        label='2nd term')
        if term_number > sd.SECOND_TERM:
            plt.scatter((x_test_array[term_number][fold_num])[:, 4], y_test_array[term_number][fold_num], color='b',
                        label='3rd term')

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
    predictions.to_csv(RESULTS_FOLDER + PREDICTION_OUTPUT_PREFIX + str(term_number) + '.csv', index=False)


if __name__ == "__main__":
    get_training_testing()

    gbt_predict(sd.FIRST_TERM, criterion='mae', learning_rate=0.01, loss='deviance', max_depth=1, max_features='log2',
                min_samples_leaf=0.1, min_samples_split=0.1, n_estimators=300, subsample=0.5)  # tuned
    # TODO: tune parameters
    gbt_predict(sd.SECOND_TERM, criterion='mae', learning_rate=0.01, loss='deviance', max_depth=1, max_features='log2',
                min_samples_leaf=0.2, min_samples_split=0.1, n_estimators=500, subsample=0.5)
    gbt_predict(sd.THIRD_TERM, 100, 0.01, 1, 0.1, 0.1, 0.5)
