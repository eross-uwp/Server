"""
___authors___: Evan Majerus & Austin FitzGerald
"""
import os

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold

__DATA_FOLDER = 'data\\ImmediatePrereqTables\\'
__FOLDS_OUTPUT = 'data\\ImmediatePrereqFolds\\'
__TRAIN_PREFIX = 'train_'
__TEST_PREFIX = 'test_'
__NUMBER_FOLDS = 5
__RANDOM_SEED = 313131
np.random.seed(__RANDOM_SEED)


def get_prereq_table(filename):
    file = pd.read_csv(__DATA_FOLDER + '\\' + filename)
    y = file.iloc[:, 1]
    x = file.drop(file.columns[1], axis=1)

    return x, y


def stratify_and_split(filename):
    x_trains = [[], [], [], [], []]
    x_tests = [[], [], [], [], []]
    y_trains = [[], [], [], [], []]
    y_tests = [[], [], [], [], []]

    x, y = get_prereq_table(filename)
    x = x.values
    y = y.values

    if len(x) >= 25 and len(y) >= 25:
        skf = StratifiedKFold(n_splits=__NUMBER_FOLDS, shuffle=True, random_state=__RANDOM_SEED)
        for train_index, test_index in skf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            x_trains.append(x_train)
            x_tests.append(x_test)
            y_trains.append(y_train)
            y_tests.append(y_test)

    return x_trains, x_tests, y_trains, y_tests

def lr_predict(postreq_name, x_train, x_test, y_train, y_test):
    model = LinearRegression()

    for fold_num in range(0, __NUMBER_FOLDS):
        model.fit(x_train[fold_num], y_train[fold_num])
        y_pred = model.predict(x_test[fold_num])

if __name__ == "__main__":
    for filename in os.listdir(__DATA_FOLDER):
        x_train, x_test, y_train, y_test = stratify_and_split(filename)
        lr_predict(filename[:-4], x_train, x_test, y_train, y_test)