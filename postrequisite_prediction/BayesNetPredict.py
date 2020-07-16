"""
___authors___: Chris Kott
"""
import enum
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from scipy.stats import loguniform
from sklearn import metrics
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import NuSVR
from sklearn.utils import column_or_1d
from bayesian_network.Summer_2020 import bn_interface

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

__data_folder = Path('data/BayesNetTables')
__folds_folder = Path('data/ALLPrereqFolds')
__results_folder = Path('results/BayesNet')
__tuning_results_folder = Path('TuningResults/BayesNetCPTs')


__NUMBER_FOLDS = 5
__RANDOM_SEED = np.int64(313131)
__MIN_SAMPLES_FOR_PREDICTING = 25

np.random.seed(__RANDOM_SEED)


def get_prereq_table(filename):
    file = pd.read_csv(__data_folder / filename)
    y = file.iloc[:, 1]
    ids = file['student_id']
    x = file.drop([file.columns[1], file.columns[0]], axis=1)  # drop the postreq grade and student_id columns
    # x = x.drop(x.columns[len(x.columns) - 1], axis=1)  # drop the term diff column
    # x = x.drop([x.columns[len(x.columns) - 1], x.columns[len(x.columns) - 2], x.columns[len(x.columns) - 3],
    #            x.columns[len(x.columns) - 4]], axis=1)  # remove all but prereqs.
    return x, y, ids


if __name__ == "__main__":
    if not os.path.exists(__tuning_results_folder):
        os.makedirs(__tuning_results_folder)

    for course in os.listdir(__data_folder):
        print(course)
        x_trains = []
        x_tests = []
        y_trains = []
        y_tests = []
        id_tests = []

        x, y, ids = get_prereq_table(course)
        x = x.fillna(-1)
        y = y.fillna(-1)
        x_columns = list(x.columns.values)
        x = x.values
        y = y.values
        if len(x) >= __MIN_SAMPLES_FOR_PREDICTING and len(y) >= __MIN_SAMPLES_FOR_PREDICTING:
            skf = StratifiedKFold(n_splits=__NUMBER_FOLDS, shuffle=True, random_state=__RANDOM_SEED)
            loop_count = 1
            for train_index, test_index in skf.split(x, y):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                da = pd.concat([pd.DataFrame(x_train), pd.DataFrame(y_train)], axis=1).astype(pd.Int64Dtype())\
                    .astype(str).replace({'<NA>': 'nan'})
                print(da)
                if os.path.exists(__tuning_results_folder/(course[:-4] + "_CPT_" + str(loop_count) + ".csv")):
                    df_cpt = pd.read_csv(__tuning_results_folder/(course[:-4] + "_CPT_" + str(loop_count) + ".csv"))
                else:
                    df_cpt = bn_interface.create_navg_cpt(da)
                    df_cpt.to_csv(__tuning_results_folder/(course[:-4] + "_CPT_" + str(loop_count) + ".csv"))
                model = bn_interface.create_bayesian_network(da, df_cpt=df_cpt)
                print()
                loop_count += 1
