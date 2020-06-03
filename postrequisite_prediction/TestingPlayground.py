import os
import sys

import pandas as pd
import numpy as np
from pip._internal.utils.misc import enum
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
import warnings


__data_folder = 'data\\AllPrereqTables\\'
__folds_folder = 'data\\Testing\\'
__model_enum = 0
__tree_type = 0
__tuning_results_folder = 'TuningResults\\All\\GBT\\'

__TRAIN_PREFIX = 'train_'
__TEST_PREFIX = 'test_'
__NUMBER_FOLDS = 5
__RANDOM_SEED = 313131
__MIN_SAMPLES_FOR_PREDICTING = 25
__MODEL_TYPES_ENUM = enum(LOGISTIC_REGRESSION=1, GRADIENT_BOOSTED_TREES=2)
__TREE_TYPES_ENUM = enum(ROOT_PREREQS=1, IMMEDIATE_PREREQS=2, ALL_PREREQS=3)


def get_prereq_table(filename):
    file = pd.read_csv(__data_folder + '\\' + filename)  # Grab files and
    y = file.iloc[:, 1]  # Grab the second row of the files
    x = file.drop([file.columns[1], file.columns[0]], axis=1)  # drop the postreq grade and student_id columns
    x = x.drop(x.columns[len(x.columns) - 1], axis=1)  # drop the term diff column
    return x, y


def stratify_and_split(filename):
    x_trains = []
    x_tests = []
    y_trains = []
    y_tests = []

    x, y = get_prereq_table(filename)
    x = x.fillna(-1)
    print(x)
    y = y.fillna(-1)
    x_columns = list(x.columns.values)
    # print(x_columns)
    x = x.values
    y = y.values
    print(x)

    if not os.path.exists(__folds_folder):
        os.makedirs(__folds_folder)

    if len(x) >= __MIN_SAMPLES_FOR_PREDICTING and len(y) >= __MIN_SAMPLES_FOR_PREDICTING:
        skf = StratifiedKFold(n_splits=__NUMBER_FOLDS, shuffle=True, random_state=__RANDOM_SEED)
        loop_count = 0
        for train_index, test_index in skf.split(x, y):  # Grabs indices to stratify x data on y column.
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            x_trains.append(x_train)
            x_tests.append(x_test)
            y_trains.append(y_train)
            y_tests.append(y_test)

            (pd.concat(
                [pd.DataFrame(x_train, columns=x_columns),
                 pd.DataFrame(y_train, columns=[filename[:-4]])],
                axis=1)).to_csv(__folds_folder + filename[:-4] + '_' +
                                __TRAIN_PREFIX + str(loop_count + 1) + '.csv', encoding='utf-8', index=False)

            (pd.concat(
                [pd.DataFrame(x_test, columns=x_columns),
                 pd.DataFrame(y_test, columns=[filename[:-4]])],
                axis=1)).to_csv(__folds_folder + filename[:-4] + '_' +
                                __TEST_PREFIX + str(loop_count + 1) + '.csv', encoding='utf-8', index=False)
            loop_count += 1

    return x_trains, x_tests, y_trains, y_tests, x_columns, len(x)


if __name__ == "__main__":
    for filename in os.listdir(__tuning_results_folder):
        filename = str(filename[:-4] + '.csv')
        stratify_and_split(filename)
