"""
___authors___: Austin FitzGerald
"""

import os
import sys

import pandas as pd
import numpy as np
from pip._internal.utils.misc import enum
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, RandomizedSearchCV
import warnings
import time
from pathlib import Path
from joblib import Parallel, delayed, parallel_backend
import pickle

from sklearn.utils import column_or_1d

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

__data_folder = Path()
__folds_folder = Path()
__results_folder = Path()
__tuning_results_folder = Path()
__model_output = Path()

__model_enum = 0
__tree_type = 0

__TRAIN_PREFIX = 'train_'
__TEST_PREFIX = 'test_'
__NUMBER_FOLDS = 5
__RANDOM_SEED = 313131
__MIN_SAMPLES_FOR_PREDICTING = 25
__MODEL_TYPES_ENUM = enum(LOGISTIC_REGRESSION=1, GRADIENT_BOOSTED_TREES=2)
__TREE_TYPES_ENUM = enum(ROOT_PREREQS=1, IMMEDIATE_PREREQS=2, ALL_PREREQS=3)

np.random.seed(__RANDOM_SEED)

flatten = lambda l: [item for sublist in l for item in
                     sublist]  # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists


def set_paths():
    if __tree_type == __TREE_TYPES_ENUM.ALL_PREREQS:
        data_folder = Path('data/AllPrereqTables/')
        folds_output = Path('data/AllPrereqFolds/')
        if __model_enum == __MODEL_TYPES_ENUM.LOGISTIC_REGRESSION:
            results_folder = Path('results/AllPrereq_LogisticRegression_Results/')
            tuning_results_folder = Path('TuningResults/All/LR/')
            model_output = Path('models/LR_model_all/')
        elif __model_enum == __MODEL_TYPES_ENUM.GRADIENT_BOOSTED_TREES:
            results_folder = Path('results/AllPrereq_GBTClassifier_Results/')
            tuning_results_folder = Path('TuningResults/All/GBT/')
            model_output = Path('models/GBT_model_all/')
    elif __tree_type == __TREE_TYPES_ENUM.ROOT_PREREQS:
        data_folder = Path('data/RootPrereqTables/')
        folds_output = Path('data/RootPrereqFolds/')
        if __model_enum == __MODEL_TYPES_ENUM.LOGISTIC_REGRESSION:
            results_folder = Path('results/RootPrereq_LogisticRegression_Results/')
            tuning_results_folder = Path('TuningResults/Root/LR/')
            model_output = Path('models/LR_model_root/')
        elif __model_enum == __MODEL_TYPES_ENUM.GRADIENT_BOOSTED_TREES:
            results_folder = Path('results/RootPrereq_GBTClassifier_Results/')
            tuning_results_folder = Path('TuningResults/Root/GBT/')
            model_output = Path('models/GBT_model_root/')
    elif __tree_type == __TREE_TYPES_ENUM.IMMEDIATE_PREREQS:
        data_folder = Path('data/ImmediatePrereqTables/')
        folds_output = Path('data/ImmediatePrereqFolds/')
        if __model_enum == __MODEL_TYPES_ENUM.LOGISTIC_REGRESSION:
            results_folder = Path('results/ImmediatePrereq_LogisticRegression_Results/')
            tuning_results_folder = Path('TuningResults/Immediate/LR/')
            model_output = Path('models/LR_model_imme/')
        elif __model_enum == __MODEL_TYPES_ENUM.GRADIENT_BOOSTED_TREES:
            results_folder = Path('results/ImmediatePrereq_GBTClassifier_Results/')
            tuning_results_folder = Path('TuningResults/Immediate/GBT/')
            model_output = Path('models/GBT_model_imme/')

    return data_folder, folds_output, results_folder, tuning_results_folder, model_output


def get_prereq_table(filename):
    file = pd.read_csv(__data_folder / filename)
    y = file.iloc[:, 1]
    x = file.drop([file.columns[1], file.columns[0]], axis=1)  # drop the postreq grade and student_id columns
    x = x.drop(x.columns[len(x.columns) - 1], axis=1)  # drop the term diff column
    return x, y


# Automatic Sequenced tuning based on:
# https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
# Note this was abandoned and is only here for legacy purposes. Will be removed in future commits. Logistic Regression
# was never touched
def tune(filename):
    loop_time = time.time()

    x, y = get_prereq_table(filename)
    x = x.fillna(-1).values
    y = y.fillna(-1).values
    if len(x) >= __MIN_SAMPLES_FOR_PREDICTING and len(y) >= __MIN_SAMPLES_FOR_PREDICTING:

        check_y = column_or_1d(y)
        unique_y, y_inversed = np.unique(check_y, return_inverse=True)
        y_counts = np.bincount(y_inversed)
        if not np.all([__NUMBER_FOLDS] > y_counts):
            # Round 1
            if __model_enum == __MODEL_TYPES_ENUM.LOGISTIC_REGRESSION:
                params = {}
                model = LogisticRegression(random_state=__RANDOM_SEED, multi_class='auto')
                param_grid = [
                    {'penalty': ['l1'], 'solver': ['liblinear', 'saga'], "C": np.logspace(-5, 8, 15)},
                    {'penalty': ['l2', 'none'], 'solver': ['newton-cg', 'sag', 'saga', 'lbfgs'], "C": np.logspace(-5, 8, 15)}
                ]
            elif __model_enum == __MODEL_TYPES_ENUM.GRADIENT_BOOSTED_TREES:
                scoring = "neg_root_mean_squared_error"
                # scoring = None
                params = {
                    "max_features": "sqrt",
                    "subsample": 0.8
                }
                model = GradientBoostingClassifier(random_state=__RANDOM_SEED, **params)
                param_grid = {
                    "learning_rate": list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 50)),
                    "n_estimators": range(10, 501, 10)
                }

            skf = StratifiedKFold(n_splits=__NUMBER_FOLDS, shuffle=True, random_state=__RANDOM_SEED)
            clf = GridSearchCV(model, param_grid, cv=skf, scoring=scoring)
            clf.fit(x, y)
            params.update(clf.best_params_)
            print(str(filename) + ": " + str(params))
            print(clf.best_score_)

            # Round 2
            if __model_enum == __MODEL_TYPES_ENUM.GRADIENT_BOOSTED_TREES:
                model = GradientBoostingClassifier(random_state=__RANDOM_SEED, **params)
                param_grid = {
                    "max_depth": range(3, 15, 1),
                    "min_samples_split": range(1, len(y), 1)
                }
                skf = StratifiedKFold(n_splits=__NUMBER_FOLDS, shuffle=True, random_state=__RANDOM_SEED)
                clf = GridSearchCV(model, param_grid, cv=skf, scoring=scoring)
                clf.fit(x, y)
            params.update(clf.best_params_)
            print(str(filename) + ": " + str(params))
            print(clf.best_score_)

            # Round 3
            if __model_enum == __MODEL_TYPES_ENUM.GRADIENT_BOOSTED_TREES:
                model = GradientBoostingClassifier(random_state=__RANDOM_SEED, **params)
                param_grid = {
                    "min_samples_leaf": range(1, len(y), 1)
                }
                skf = StratifiedKFold(n_splits=__NUMBER_FOLDS, shuffle=True, random_state=__RANDOM_SEED)
                clf = GridSearchCV(model, param_grid, cv=skf, scoring=scoring)
                clf.fit(x, y)
            params.update(clf.best_params_)
            print(str(filename) + ": " + str(params))
            print(clf.best_score_)

            # Round 4
            if __model_enum == __MODEL_TYPES_ENUM.GRADIENT_BOOSTED_TREES:
                model = GradientBoostingClassifier(random_state=__RANDOM_SEED, **params)
                param_grid = {"max_features": ['log2', 'sqrt']}
                '''[
                    {"max_features": range(1, x.shape[1], 1)},
                    {"max_features": ['log2', 'sqrt']}
                    ]'''
                skf = StratifiedKFold(n_splits=__NUMBER_FOLDS, shuffle=True, random_state=__RANDOM_SEED)
                clf = GridSearchCV(model, param_grid, cv=skf, scoring=scoring)
                clf.fit(x, y)
            params.update(clf.best_params_)
            print(str(filename) + ": " + str(params))
            print(clf.best_score_)

            # Round 5
            if __model_enum == __MODEL_TYPES_ENUM.GRADIENT_BOOSTED_TREES:
                model = GradientBoostingClassifier(random_state=__RANDOM_SEED, **params)
                param_grid = {
                    "loss": ['deviance', 'exponential'],
                    'subsample': np.arange(0.1, 1.1, 0.1),
                    'criterion': ['friedman_mse', 'mse', 'mae']
                }
                skf = StratifiedKFold(n_splits=__NUMBER_FOLDS, shuffle=True, random_state=__RANDOM_SEED)
                clf = GridSearchCV(model, param_grid, cv=skf, scoring=scoring)
                clf.fit(x, y)
            params.update(clf.best_params_)

            np.save(__tuning_results_folder / filename[:-4], params)
            print(filename[:-4] + " " + str(round(time.time() - loop_time, 2)) + "s.: " + str(params))
            print(clf.best_score_)
            print()


# Grid based exhaustive search tuning on small amount of parameters. Based on the existing code.
def tune_grid(filename):
    loop_time = time.time()
    scoring = None
    if __model_enum == __MODEL_TYPES_ENUM.LOGISTIC_REGRESSION:
        scoring = "neg_root_mean_squared_error"
        model = LogisticRegression(random_state=__RANDOM_SEED)
        c_space = list(np.logspace(-7, 7, 60))
        param_grid = [
            {'penalty': ['l1', 'l2'], 'solver': ['liblinear'], "C": c_space, "class_weight": ['balanced', None]},
            {'penalty': ['l2', 'none'], 'solver': ['newton-cg', 'sag', 'lbfgs'], "C": c_space,
             "class_weight": ['balanced', None]},
            {'penalty': ['l1', 'l2', 'none', 'elasticnet'], 'solver': ['saga'], "C": c_space,
             "class_weight": ['balanced', None]}
        ]
    elif __model_enum == __MODEL_TYPES_ENUM.GRADIENT_BOOSTED_TREES:
        scoring = "neg_root_mean_squared_error"
        model = GradientBoostingClassifier(random_state=__RANDOM_SEED)
        param_grid = {
            "loss": ["deviance"],
            "learning_rate": [0.01, 1],
            "min_samples_split": [0.1, 0.5],
            "min_samples_leaf": [0.1, 0.5],
            "max_depth": [1, 4],
            "max_features": ["log2", "sqrt"],
            "criterion": ["friedman_mse", "mae"],
            "subsample": [0.8],
            "n_estimators": [300, 500]
        }

    skf = StratifiedKFold(n_splits=__NUMBER_FOLDS, shuffle=True, random_state=__RANDOM_SEED)
    clf = GridSearchCV(model, param_grid, cv=skf, scoring=scoring, verbose=True)

    x, y = get_prereq_table(filename)
    x = x.fillna(-1).values
    y = y.fillna(-1).values
    if len(x) >= __MIN_SAMPLES_FOR_PREDICTING and len(y) >= __MIN_SAMPLES_FOR_PREDICTING:
        check_y = column_or_1d(y)
        unique_y, y_inversed = np.unique(check_y, return_inverse=True)
        y_counts = np.bincount(y_inversed)
        if not np.all([__NUMBER_FOLDS] > y_counts):
            best_clf = clf.fit(x, y)

            np.save(__tuning_results_folder / filename[:-4], best_clf.best_params_)
            print(filename[:-4] + " " + str(round(time.time() - loop_time, 2)) + "s.: " + str(best_clf.best_score_))
            print(best_clf.best_params_)
            print()


# Random tuning based on extended parameter grid. Preferred method at the moment based on:
# https://medium.com/rants-on-machine-learning/smarter-parameter-sweeps-or-why-grid-search-is-plain-stupid-c17d97a0e881
def tune_rand(filename):
    loop_time = time.time()
    scoring = None
    x, y = get_prereq_table(filename)
    if __model_enum == __MODEL_TYPES_ENUM.LOGISTIC_REGRESSION:
        model = LogisticRegression(random_state=__RANDOM_SEED)
        c_space = list(np.logspace(-7, 7, 60))
        param_grid = [
            {'penalty': ['l1', 'l2'], 'solver': ['liblinear'], "C": c_space, "class_weight": ['balanced', None]},
            {'penalty': ['l2', 'none'], 'solver': ['newton-cg', 'sag', 'lbfgs'], "C": c_space,
             "class_weight": ['balanced', None]},
            {'penalty': ['l1', 'l2', 'none', 'elasticnet'], 'solver': ['saga'], "C": c_space,
             "class_weight": ['balanced', None]}
        ]
    elif __model_enum == __MODEL_TYPES_ENUM.GRADIENT_BOOSTED_TREES:
        scoring = "neg_root_mean_squared_error"
        model = GradientBoostingClassifier(random_state=__RANDOM_SEED)
        param_grid = {
            "loss": ["deviance", "exponential"],
            "learning_rate": list(np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)),
            "min_samples_split": list(range(1, len(y), 1)),
            "min_samples_leaf": list(range(1, len(y), 1)),
            "max_depth": list(range(2, 26, 1)),
            "max_features": ["log2", "sqrt"],
            "criterion": ["friedman_mse", "mae", "mse"],
            "subsample": list(np.arange(0.1, 1.1, 0.1)),
            "n_estimators": list(range(10, 1501, 10))
        }

    skf = StratifiedKFold(n_splits=__NUMBER_FOLDS, shuffle=True, random_state=__RANDOM_SEED)
    clf = RandomizedSearchCV(model, param_grid, cv=skf, scoring=scoring, n_iter=2500, random_state=__RANDOM_SEED, verbose=True)

    x = x.fillna(-1).values
    y = y.fillna(-1).values
    if len(x) >= __MIN_SAMPLES_FOR_PREDICTING and len(y) >= __MIN_SAMPLES_FOR_PREDICTING:
        check_y = column_or_1d(y)
        unique_y, y_inversed = np.unique(check_y, return_inverse=True)
        y_counts = np.bincount(y_inversed)
        if not np.all([__NUMBER_FOLDS] > y_counts):
            best_clf = clf.fit(x, y)
            np.save(__tuning_results_folder / filename[:-4], best_clf.best_params_)
            print(filename[:-4] + " " + str(round(time.time() - loop_time, 2)) + "s.: " + str(best_clf.best_score_))
            print(best_clf.best_params_)
            print()


def hyperparameter_tuning():
    print('Hyperparameter tuning beginning. Run time will print after the completion of each tuning. \n')

    start_time = time.time()
    if not os.path.exists(__tuning_results_folder):
        os.makedirs(__tuning_results_folder)

    with parallel_backend('loky', n_jobs=-1):
        for filename in sorted(os.listdir(__data_folder)):
            # tune(filename)
            tune_grid(filename)
            # tune_rand(filename)

    print('Hyperparameter tuning completed in ' + str(round(time.time() - start_time, 2)) + 's. Files saved to: \''
          + str(__tuning_results_folder) + '\' \n')


def reverse_convert_grade(int_grade):
    if int_grade == 10:
        return 'A'
    elif int_grade == 9:
        return 'A-'
    elif int_grade == 8:
        return 'B+'
    elif int_grade == 7:
        return 'B'
    elif int_grade == 6:
        return 'B-'
    elif int_grade == 5:
        return 'C+'
    elif int_grade == 4:
        return 'C'
    elif int_grade == 3:
        return 'C-'
    elif int_grade == 2:
        return 'D+'
    elif int_grade == 1:
        return 'D'
    elif int_grade == 0:
        return 'F'


def predict(postreq_name, x_train, x_test, y_train, y_test, x_columns):
    if not os.path.exists(__tuning_results_folder / (postreq_name + '.npy')):
        read_dictionary = None
    else:
        read_dictionary = np.load(__tuning_results_folder / (postreq_name + '.npy'), allow_pickle=True).item()

    print(postreq_name + " Parameter Dictionary: " + str(read_dictionary))

    if __model_enum == __MODEL_TYPES_ENUM.LOGISTIC_REGRESSION:
        if read_dictionary is None:
            model = LogisticRegression(random_state=__RANDOM_SEED)
        else:
            model = LogisticRegression(random_state=__RANDOM_SEED, **read_dictionary)
    elif __model_enum == __MODEL_TYPES_ENUM.GRADIENT_BOOSTED_TREES:
        if read_dictionary is None:
            model = GradientBoostingClassifier(random_state=__RANDOM_SEED)
        else:
            model = GradientBoostingClassifier(random_state=__RANDOM_SEED, **read_dictionary)

    y_preds = []
    #   F,  D,  D+, C-, C,  C+, B-, B,  B+, A-, A
    y_grades = [[], [], [], [], [], [], [], [], [], [], []]

    for fold_num in range(0, __NUMBER_FOLDS):
        # print(x_train[fold_num])
        model.fit(x_train[fold_num], y_train[fold_num])
        y_pred = model.predict(x_test[fold_num])
        y_preds += list(y_pred)

        temp = model.predict_proba(x_test[fold_num])

        for t in temp:
            count = 0
            not_filled = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            for f in t:
                y_grades[model.classes_[count]].append(f)
                not_filled.remove(model.classes_[count])
                count += 1
            for q in not_filled:
                y_grades[q].append(0)

    rr = metrics.r2_score(flatten(y_test), y_preds)
    rmse = metrics.mean_squared_error(flatten(y_test), y_preds, squared=False)
    acc = metrics.accuracy_score(flatten(y_test), y_preds)

    with open(__results_folder / (postreq_name + '.txt'), "w") as text_file:
        text_file.write(
            'R^2 = ' + str(rr) + ', Accuracy = ' + str(acc) + ' , RMSE = ' + str(rmse) + ', NRMSE = ' + str(rmse / 10))

    x_df = pd.concat([pd.DataFrame(x_test[0]),
                      pd.DataFrame(x_test[1]),
                      pd.DataFrame(x_test[2]),
                      pd.DataFrame(x_test[3]),
                      pd.DataFrame(x_test[4])], ignore_index=True)
    x_df.columns = x_columns
    x_df['struggle'] = x_df['struggle'].replace(3, 'G')
    x_df['struggle'] = x_df['struggle'].replace(2, 'S')
    x_df['struggle'] = x_df['struggle'].replace(1, 'E')

    y_df = pd.concat([pd.DataFrame(y_test[0]),
                      pd.DataFrame(y_test[1]),
                      pd.DataFrame(y_test[2]),
                      pd.DataFrame(y_test[3]),
                      pd.DataFrame(y_test[4])], ignore_index=True)
    y_df.columns = [postreq_name]
    y_df[postreq_name] = y_df[postreq_name].replace(0, 'F')
    y_df[postreq_name] = y_df[postreq_name].replace(1, 'D')
    y_df[postreq_name] = y_df[postreq_name].replace(2, 'D+')
    y_df[postreq_name] = y_df[postreq_name].replace(3, 'C-')
    y_df[postreq_name] = y_df[postreq_name].replace(4, 'C')
    y_df[postreq_name] = y_df[postreq_name].replace(5, 'C+')
    y_df[postreq_name] = y_df[postreq_name].replace(6, 'B-')
    y_df[postreq_name] = y_df[postreq_name].replace(7, 'B')
    y_df[postreq_name] = y_df[postreq_name].replace(8, 'B+')
    y_df[postreq_name] = y_df[postreq_name].replace(9, 'A-')
    y_df[postreq_name] = y_df[postreq_name].replace(10, 'A')

    converted_y_preds = []
    for yp in y_preds:
        converted_y_preds.append(reverse_convert_grade(yp))
    y_predict_df = pd.DataFrame(converted_y_preds, columns=['predicted score'])

    y_grades_df = pd.DataFrame(
        {'F': y_grades[0], 'D': y_grades[1], 'D+': y_grades[2], 'C-': y_grades[3], 'C': y_grades[4],
         'C+': y_grades[5], 'B-': y_grades[6], 'B': y_grades[7], 'B+': y_grades[8],
         'A-': y_grades[9], 'A': y_grades[10]})
    predictions = pd.concat([x_df, y_df, y_predict_df, y_grades_df], axis=1)
    predictions.to_csv(__results_folder / ('PREDICTION_' + postreq_name + '.csv'), index=False)

    return predictions['predicted score'].values, y_df[postreq_name].values, rr, acc, (rmse / 10), model


def stratify_and_split(filename):
    x_trains = []
    x_tests = []
    y_trains = []
    y_tests = []

    x, y = get_prereq_table(filename)
    x = x.fillna(-1)
    y = y.fillna(-1)
    x_columns = list(x.columns.values)
    x = x.values
    y = y.values

    if len(x) >= __MIN_SAMPLES_FOR_PREDICTING and len(y) >= __MIN_SAMPLES_FOR_PREDICTING:
        skf = StratifiedKFold(n_splits=__NUMBER_FOLDS, shuffle=True, random_state=__RANDOM_SEED)
        loop_count = 0
        for train_index, test_index in skf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            x_trains.append(x_train)
            x_tests.append(x_test)
            y_trains.append(y_train)
            y_tests.append(y_test)

            (pd.concat(
                [pd.DataFrame(x_train, columns=x_columns),
                 pd.DataFrame(y_train, columns=[filename[:-4]])],
                axis=1)).to_csv(__folds_folder / (filename[:-4] + '_' +
                                __TRAIN_PREFIX + str(loop_count + 1) + '.csv'), encoding='utf-8', index=False)

            (pd.concat(
                [pd.DataFrame(x_test, columns=x_columns),
                 pd.DataFrame(y_test, columns=[filename[:-4]])],
                axis=1)).to_csv(__folds_folder / (filename[:-4] + '_' +
                                __TEST_PREFIX + str(loop_count + 1) + '.csv'), encoding='utf-8', index=False)
            loop_count += 1

    return x_trains, x_tests, y_trains, y_tests, x_columns, len(x)


def read_predict_write():
    print('Training and testing beginning. A counter will print after the completion of each training set. \n')
    if not os.path.exists(__folds_folder):
        os.makedirs(__folds_folder)
    if not os.path.exists(__results_folder):
        os.makedirs(__results_folder)

    big_predicted = []
    big_actual = []

    results_each_postreq = [[], [], [], [], []]

    counter = 0
    for filename in sorted(os.listdir(__data_folder)):
        filename = str(filename[:-4] + '.csv')
        x_train, x_test, y_train, y_test, x_columns, n_samples = stratify_and_split(filename)
        if n_samples > __MIN_SAMPLES_FOR_PREDICTING:
            predicted, actual, rr, acc, nrmse, model = predict(filename[:-4], x_train, x_test, y_train, y_test, x_columns)

            big_predicted += list(predicted)
            big_actual += list(actual)
            results_each_postreq[0].append(filename[:-4])
            results_each_postreq[1].append(rr)
            results_each_postreq[2].append(acc)
            results_each_postreq[3].append(nrmse)
            results_each_postreq[4].append(n_samples)
            print(counter)
            counter += 1

    predictions = pd.DataFrame(big_predicted, columns=['predicted'])
    actuals = pd.DataFrame(big_actual, columns=['actual'])
    all_results = pd.concat([predictions, actuals], axis=1)
    all_results.to_csv(__results_folder / 'ALL_COURSES_PREDICTIONS.csv', index=False)

    all_stats = pd.DataFrame(
        {'postreq': results_each_postreq[0], 'r^2': results_each_postreq[1], 'accuracy': results_each_postreq[2],
         'nrmse': results_each_postreq[3], 'n': results_each_postreq[4]})
    all_stats.to_csv(__results_folder / 'ALL_COURSES_STATS.csv', index=False)

    print('Model training, testing, and evaluation completed. Files saved to: \'' + str(__results_folder) + '\' \n')


def save_models():
    print('Model saving beginning. \n')
    start_time = time.time()
    if not os.path.exists(__model_output):
        os.makedirs(__model_output)

    with parallel_backend('loky', n_jobs=-1):
        Parallel()(delayed(dump_model)(filename) for filename in os.listdir(__tuning_results_folder))

    print('Model saving completed in ' + str(round(time.time()-start_time, 2)) + 's. Files saved to: '
          + str(__model_output) + '\n')


def dump_model(filename):
    filename = str(filename[:-4] + '.csv')
    x, y = get_prereq_table(filename)
    x_columns = list(x.columns.values)
    x = x.fillna(-1).values
    y = y.fillna(-1).values

    read_dictionary = np.load(__tuning_results_folder / (filename[:-4] + '.npy'), allow_pickle=True).item()

    if __model_enum == __MODEL_TYPES_ENUM.LOGISTIC_REGRESSION:
        model = LogisticRegression(random_state=__RANDOM_SEED, **read_dictionary)
    elif __model_enum == __MODEL_TYPES_ENUM.GRADIENT_BOOSTED_TREES:
        model = GradientBoostingClassifier(random_state=__RANDOM_SEED, **read_dictionary)

    model.fit(x, y)

    pickle.dump(model, open(__model_output / (filename[:-4] + '.pkl'), 'wb'))


if __name__ == "__main__":
    __tree_type = int(input("Enter one of the following for prereq type: \n"
                            "'1': Root prerequisites \n"
                            "'2': Immediate prerequisites \n"
                            "'3': All prerequisites \n"))

    if __tree_type != __TREE_TYPES_ENUM.ROOT_PREREQS and __tree_type != __TREE_TYPES_ENUM.IMMEDIATE_PREREQS and __tree_type != __TREE_TYPES_ENUM.ALL_PREREQS:
        raise ValueError('An invalid tree type was passed. Must be \'1\', \'2\', or \'3\'')

    __model_enum = int(input("Enter one of the following for model type: \n"
                             "'1': Logistic Regression \n"
                             "'2': Gradient Boosted Trees Classifier \n"))

    if __model_enum != __MODEL_TYPES_ENUM.LOGISTIC_REGRESSION and __model_enum != __MODEL_TYPES_ENUM.GRADIENT_BOOSTED_TREES:
        raise ValueError('An invalid model type was passed. Must be \'1\' or \'2\'')

    __data_folder, __folds_folder, __results_folder, __tuning_results_folder, __model_output = set_paths()

    tune_or_predict = int(input("Enter one of the following process types: \n"
                                "'1': Tune hyperparameters \n"
                                "'2': Run predictions \n"
                                "'3': Save models \n"))

    if tune_or_predict != 1 and tune_or_predict != 2 and tune_or_predict != 3:
        raise ValueError('An invalid process type was passed. Must be \'1\', \'2\', or \'3\'')

    if tune_or_predict == 1:
        hyperparameter_tuning()
    elif tune_or_predict == 2:
        read_predict_write()
    elif tune_or_predict == 3:
        save_models()
