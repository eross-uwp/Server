import enum
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn import metrics
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, \
    RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest, chi2, f_regression, f_classif, mutual_info_classif, \
    mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.utils import column_or_1d

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses


__data_folder = Path('data/AllPrereqTables/')
__data_folder1 = Path('data/ImmediatePrereqTables/')
__data_folder2 = Path('data/RootPrereqTables/')
__data_folders = [__data_folder, __data_folder1, __data_folder2]
__folds_folder = Path('data/Testing/')
__model_enum = 0
__tree_type = 0
__tuning_results_folder = Path('TuningResults/ALL/RANDOM_FOREST_REGRESSOR/')
__results_folder = Path('results/')

__TRAIN_PREFIX = 'train_'
__TEST_PREFIX = 'test_'
__NUMBER_FOLDS = 5
__RANDOM_SEED = 313131
__MIN_SAMPLES_FOR_PREDICTING = 25
__MODEL_TYPES_ENUM = enum.IntEnum('__MODEL_TYPES_ENUM', 'LOGISTIC_REGRESSION GBT_CLASSIFIER NU_SVR')


# https://stackoverflow.com/a/43886290
def round_school(x):
    if x < 0:
        return 0
    if x > 10:
        return 10
    else:
        i, f = divmod(x, 1)
        return int(i + ((f >= 0.5) if (x > 0) else (f > 0.5)))


def rounding_rmse_scorer(y, y_pred):
    return -metrics.mean_squared_error(y, [round_school(num) for num in y_pred], squared=False)


def get_prereq_table(filename):
    file = pd.read_csv(__data_folder / filename)
    y = file.iloc[:, 1]
    ids = file['student_id']
    x = file.drop([file.columns[1], file.columns[0]], axis=1)  # drop the postreq grade and student_id columns
    x = x.drop(x.columns[len(x.columns) - 1], axis=1)  # drop the term diff column
    return x, y, ids


def model_trials():
    with parallel_backend('loky', n_jobs=-1):
        for filename in sorted(os.listdir(__data_folder)):
            print(filename[:-4])
            loop_time = time.time()
            scoring = make_scorer(rounding_rmse_scorer)
            x, y, _ = get_prereq_table(filename)
            pipe = Pipeline([('classifier', DummyClassifier())])
            search_space = {'classifier': [LogisticRegression(random_state=__RANDOM_SEED),
                                           RandomForestClassifier(random_state=__RANDOM_SEED),
                                           RandomForestRegressor(random_state=__RANDOM_SEED),
                                           GradientBoostingClassifier(random_state=__RANDOM_SEED),
                                           GradientBoostingRegressor(random_state=__RANDOM_SEED),
                                           ExtraTreesClassifier(random_state=__RANDOM_SEED),
                                           ExtraTreesRegressor(random_state=__RANDOM_SEED),
                                           ExtraTreeClassifier(random_state=__RANDOM_SEED),
                                           ExtraTreeRegressor(random_state=__RANDOM_SEED),
                                           DecisionTreeClassifier(random_state=__RANDOM_SEED),
                                           DecisionTreeRegressor(random_state=__RANDOM_SEED),
                                           BernoulliNB(),
                                           GaussianNB(),
                                           LinearSVC(random_state=__RANDOM_SEED),
                                           LinearSVR(random_state=__RANDOM_SEED),
                                           NuSVR(),
                                           SVC(random_state=__RANDOM_SEED),
                                           SVR()]}
            skf = StratifiedKFold(n_splits=__NUMBER_FOLDS, shuffle=True, random_state=__RANDOM_SEED)
            clf = GridSearchCV(pipe, search_space, cv=skf, scoring=scoring, verbose=True)

            x = x.fillna(-1).values
            y = y.fillna(-1).values
            if len(x) >= __MIN_SAMPLES_FOR_PREDICTING and len(y) >= __MIN_SAMPLES_FOR_PREDICTING:
                check_y = column_or_1d(y)
                unique_y, y_inversed = np.unique(check_y, return_inverse=True)
                y_counts = np.bincount(y_inversed)
                if not np.all([__NUMBER_FOLDS] > y_counts):
                    clf.fit(x, y)
                    for i in range(len(clf.cv_results_['mean_test_score'])):
                        print(str(clf.cv_results_['mean_test_score'][i]) + ": " + str(type(clf.cv_results_['params'][i]['classifier'])))
                    print(type(clf.best_estimator_['classifier']))
                    print()


def model_selection_string():
    string = "Enter one of the following for model type: \n"
    for m_type in __MODEL_TYPES_ENUM:
        string += " '" + str(m_type.value) + "': " + m_type.name + " \n"
    return string


def dothing(i):
    return i, i ** 2


if __name__ == "__main__":
    nums = Parallel(n_jobs=-1)(delayed(dothing)(i) for i in range(50))
    print(nums)

    '''
    student_ids = []
    for datapath in __data_folders:
        for filename in os.listdir(datapath):
            _, _, ids = get_prereq_table(filename)
            student_ids.extend(ids)
    print(len(np.unique(student_ids)))
    distinctids = pd.DataFrame(np.unique(student_ids), columns=['distinct ids'])
    distinctids.to_csv(__results_folder/'distinct_ids.csv', index=False)
'''
    """
    # bestFeatures = SelectKBest(score_func=mutual_info_classif, k='all')
    # print(bestFeatures.score_func)
    for filename in os.listdir(__data_folder):
        x, y = get_prereq_table(filename)
        if not os.path.exists(__tuning_results_folder / (filename[:-4] + '.npy')):
            read_dictionary = None
        else:
            read_dictionary = np.load(__tuning_results_folder / (filename[:-4] + '.npy'), allow_pickle=True).item()
        print(filename, ":", read_dictionary)
        if read_dictionary is None:
            model = RandomForestRegressor(random_state=__RANDOM_SEED)
        else:
            model = RandomForestRegressor(**read_dictionary, random_state=__RANDOM_SEED)

        scoring = metrics.make_scorer(rounding_rmse_scorer)

        skf = StratifiedKFold(n_splits=__NUMBER_FOLDS, shuffle=True, random_state=__RANDOM_SEED)
        clf = GridSearchCV(model, {}, cv=skf, scoring=scoring, n_jobs=-1)

        x2 = x.fillna(-1).values
        y2 = y.fillna(-1).values
        if x.shape[0] > 25:
            '''
            fit = bestFeatures.fit(x2, y2)
            scores = pd.DataFrame(fit.scores_)
            labels = pd.DataFrame(x.columns)
            total = pd.concat([labels, scores], axis=1)

            total.columns = ['Feature', 'Score']
            print(filename[:-4])
            print(total)
            print()
            '''
            clf.fit(x2, y2)
            print(filename[:-4], ":", clf.best_score_)
            scores = pd.DataFrame(clf.best_estimator_.feature_importances_)
            labels = pd.DataFrame(x.columns)
            total = pd.concat([labels, scores], axis=1)
            total.columns = ['Feature', 'Score']
            print(total)
            print()
"""
