"""
___authors___: Austin FitzGerald, Chris Kott
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
__RANDOM_SEED = np.int64(313131)
__MIN_SAMPLES_FOR_PREDICTING = 25
__MODEL_TYPES_ENUM = enum.IntEnum('__MODEL_TYPES_ENUM', 'LOGISTIC_REGRESSION GBT_CLASSIFIER NU_SVR GBT_REGRESSOR '
                                                        'RANDOM_FOREST_REGRESSOR MOD_ZEROR MEAN_ZEROR')
__TREE_TYPES_ENUM = enum.IntEnum('__TREE_TYPES_ENUM', 'ROOT IMMEDIATE ALL')

np.random.seed(__RANDOM_SEED)

flatten = lambda l: [item for sublist in l for item in
                     sublist]  # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists


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


def set_paths():
    data_folder = Path('data/' + __tree_type.name + 'PrereqTables/')
    folds_output = Path('data/' + __tree_type.name + 'PrereqFolds/')
    results_folder = Path('results/' + __tree_type.name + 'Prereq_' + __model_enum.name + '_Results/')
    tuning_results_folder = Path('TuningResults/' + __tree_type.name + '/' + __model_enum.name + '/')
    model_output = Path('models/' + __model_enum.name + '_model_' + __tree_type.name + '/')
    return data_folder, folds_output, results_folder, tuning_results_folder, model_output


def get_prereq_table(filename):
    file = pd.read_csv(__data_folder / filename)
    y = file.iloc[:, 1]
    ids = file['student_id']
    x = file.drop([file.columns[1], file.columns[0]], axis=1)  # drop the postreq grade and student_id columns
    # x = x.drop(x.columns[len(x.columns) - 1], axis=1)  # drop the term diff column
    x = x.drop([x.columns[len(x.columns) - 1], x.columns[len(x.columns) - 2], x.columns[len(x.columns) - 3],
                x.columns[len(x.columns) - 4]], axis=1)  # remove all but prereqs.
    return x, y, ids


# Random tuning based on extended parameter grid. Preferred method at the moment based on:
# https://medium.com/rants-on-machine-learning/smarter-parameter-sweeps-or-why-grid-search-is-plain-stupid-c17d97a0e881
def tune_rand(filename):
    loop_time = time.time()
    x, y, _ = get_prereq_table(filename)
    rng = np.random.RandomState(__RANDOM_SEED)
    if __model_enum == __MODEL_TYPES_ENUM.LOGISTIC_REGRESSION:
        num_trials = 2400
        model = LogisticRegression(random_state=rng)
        c_space = list(np.logspace(-7, 7, 100))
        param_grid = [
            {'penalty': ['l1', 'l2'], 'solver': ['liblinear'], "C": c_space, "class_weight": ['balanced', None]},
            {'penalty': ['l2', 'none'], 'solver': ['newton-cg', 'sag', 'lbfgs'], "C": c_space,
             "class_weight": ['balanced', None]},
            {'penalty': ['l1', 'l2', 'none', 'elasticnet'], 'solver': ['saga'], "C": c_space,
             "class_weight": ['balanced', None]}
        ]
    elif __model_enum == __MODEL_TYPES_ENUM.RANDOM_FOREST_REGRESSOR:
        model = RandomForestRegressor(random_state=rng)
        num_trials = 1500
        param_grid = {
            "n_estimators": np.logspace(np.log10(10), np.log10(1500), 100, dtype='int64'),
            "criterion": ["friedman_mse", "mae", "mse"],
            "min_samples_split": list(range(1, len(y), 1)),
            "min_samples_leaf": list(range(1, len(y), 1)),
            "max_features": ["auto", "sqrt", "log2"]
        }
    elif __model_enum == __MODEL_TYPES_ENUM.GBT_CLASSIFIER:
        num_trials = 2000
        model = GradientBoostingClassifier(random_state=rng)
        param_grid = {
            "loss": ["deviance", "exponential"],
            "learning_rate": np.logspace(np.log10(0.005), np.log10(0.5), 100),
            "min_samples_split": list(range(1, len(y), 1)),
            "min_samples_leaf": list(range(1, len(y), 1)),
            "max_depth": list(range(2, 26, 1)),
            "max_features": ["log2", "sqrt"],
            "criterion": ["friedman_mse", "mae", "mse"],
            "subsample": list(np.arange(0.1, 1.1, 0.05)),
            "n_estimators": np.logspace(np.log10(10), np.log10(1500), 100, dtype='int64')
        }
    elif __model_enum == __MODEL_TYPES_ENUM.NU_SVR:
        num_trials = 2500
        model = NuSVR()
        c_space = list(np.logspace(-3, 3, 25))
        nu_space = np.arange(0.1, 1.1, 0.05)
        param_grid = [{'nu': nu_space, 'C': c_space, 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}]
    elif __model_enum == __MODEL_TYPES_ENUM.GBT_REGRESSOR:
        num_trials = 2000
        model = GradientBoostingRegressor(random_state=rng)
        param_grid = {
            "loss": ['ls', 'lad', 'huber', 'quantile'],
            "learning_rate": np.logspace(np.log10(0.005), np.log10(0.5), 100),
            "min_samples_split": list(range(1, len(y), 1)),
            "min_samples_leaf": list(range(1, len(y), 1)),
            "max_depth": list(range(2, 26, 1)),
            "max_features": ["log2", "sqrt"],
            "criterion": ["friedman_mse", "mae", "mse"],
            "subsample": list(np.arange(0.1, 1.1, 0.05)),
            "n_estimators": np.logspace(np.log10(10), np.log10(1500), 100, dtype='int64')
        }
    else:
        raise NotImplementedError("This method has not been implemented for " + str(__model_enum.name))

    scoring = metrics.make_scorer(rounding_rmse_scorer)
    skf = StratifiedKFold(n_splits=__NUMBER_FOLDS, shuffle=True, random_state=rng)
    clf = RandomizedSearchCV(model, param_grid, cv=skf, scoring=scoring, n_iter=num_trials, random_state=rng,
                             verbose=True)

    x = x.fillna(-1).values
    y = y.fillna(-1).values
    if len(x) >= __MIN_SAMPLES_FOR_PREDICTING and len(y) >= __MIN_SAMPLES_FOR_PREDICTING:
        check_y = column_or_1d(y)
        unique_y, y_inversed = np.unique(check_y, return_inverse=True)
        y_counts = np.bincount(y_inversed)
        if not np.all([__NUMBER_FOLDS] > y_counts):
            best_clf = clf.fit(x, y)

            if os.path.exists(__tuning_results_folder / (filename[:-4] + '.npy')) and False:
                old_params = np.load(__tuning_results_folder / (filename[:-4] + '.npy'), allow_pickle=True).item()
                for key in old_params:
                    old_params[key] = [old_params[key]]
                new_params = {}
                for key in best_clf.best_params_:
                    new_params[key] = [best_clf.best_params_[key]]
                print('old ', old_params)
                print('new ', new_params)
                if old_params != new_params:
                    param_grid = [new_params, old_params]
                    clf2 = GridSearchCV(model, param_grid, cv=skf, scoring=scoring, verbose=True)
                    best_clf = clf2.fit(x, y)

            np.save(__tuning_results_folder / filename[:-4], best_clf.best_params_)
            print(filename[:-4] + " " + str(round(time.time() - loop_time, 2)) + "s.: " + str(best_clf.best_score_))
            print(best_clf.best_params_)
            print()


def hyperparameter_tuning():
    if not (__model_enum == __MODEL_TYPES_ENUM.BAYESIAN_NETWORK and (
            __tree_type == __TREE_TYPES_ENUM.ALL or __tree_type == __TREE_TYPES_ENUM.IMMEDIATE)):
        print('Hyperparameter tuning beginning. Run time will print after the completion of each tuning. \n')

        start_time = time.time()
        if not os.path.exists(__tuning_results_folder):
            os.makedirs(__tuning_results_folder)

        with parallel_backend('loky', n_jobs=-1):
            for filename in sorted(os.listdir(__data_folder)):
                tune_rand(filename)
                # tune(filename)
                # tune_grid(filename)

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


def reverse_convert_struggle(int_struggle):
    if int_struggle == 1:
        return 'E'
    elif int_struggle == 2:
        return 'S'
    elif int_struggle == 3:
        return 'G'
    else:
        return '?'


def predict(postreq_name, x_train, x_test, y_train, y_test, x_columns):
    if not os.path.exists(__tuning_results_folder / (postreq_name + '.npy')):
        read_dictionary = None
    else:
        read_dictionary = np.load(__tuning_results_folder / (postreq_name + '.npy'), allow_pickle=True).item()

    print(__model_enum.name + " " + postreq_name + " Parameter Dictionary: " + str(read_dictionary))

    if __model_enum == __MODEL_TYPES_ENUM.LOGISTIC_REGRESSION:
        if read_dictionary is None:
            model = LogisticRegression(random_state=__RANDOM_SEED)
        else:
            model = LogisticRegression(random_state=__RANDOM_SEED, **read_dictionary)
    elif __model_enum == __MODEL_TYPES_ENUM.GBT_CLASSIFIER:
        if read_dictionary is None:
            model = GradientBoostingClassifier(random_state=__RANDOM_SEED)
        else:
            model = GradientBoostingClassifier(random_state=__RANDOM_SEED, **read_dictionary)
    elif __model_enum == __MODEL_TYPES_ENUM.NU_SVR:
        if read_dictionary is None:
            model = NuSVR()
        else:
            model = NuSVR(**read_dictionary)
    elif __model_enum == __MODEL_TYPES_ENUM.GBT_REGRESSOR:
        model = GradientBoostingRegressor(random_state=__RANDOM_SEED)
    elif __model_enum == __MODEL_TYPES_ENUM.RANDOM_FOREST_REGRESSOR:
        if read_dictionary is None:
            model = RandomForestRegressor(random_state=__RANDOM_SEED)
        else:
            model = RandomForestRegressor(**read_dictionary, random_state=__RANDOM_SEED)
    elif __model_enum == __MODEL_TYPES_ENUM.MOD_ZEROR:
        model = DummyClassifier('most_frequent')
    elif __model_enum == __MODEL_TYPES_ENUM.MEAN_ZEROR:
        model = DummyRegressor('mean')

    y_preds = []
    #   F,  D,  D+, C-, C,  C+, B-, B,  B+, A-, A
    y_grades = [[], [], [], [], [], [], [], [], [], [], []]

    for fold_num in range(0, __NUMBER_FOLDS):
        # print(x_train[fold_num])
        model.fit(x_train[fold_num], y_train[fold_num])
        y_pred = model.predict(x_test[fold_num])
        y_preds += list(y_pred)

        if __model_enum == __MODEL_TYPES_ENUM.LOGISTIC_REGRESSION or __model_enum == __MODEL_TYPES_ENUM.GBT_CLASSIFIER:
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

    y_preds = [round_school(num) for num in y_preds]

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
    # x_df['struggle'] = x_df['struggle'].apply(reverse_convert_struggle)

    y_df = pd.concat([pd.DataFrame(y_test[0]),
                      pd.DataFrame(y_test[1]),
                      pd.DataFrame(y_test[2]),
                      pd.DataFrame(y_test[3]),
                      pd.DataFrame(y_test[4])], ignore_index=True)
    y_df.columns = [postreq_name]
    y_df[postreq_name] = y_df[postreq_name].apply(reverse_convert_grade)

    y_predict_df = pd.DataFrame(y_preds, columns=['predicted score'])
    y_predict_df['predicted score'] = y_predict_df['predicted score'].apply(reverse_convert_grade)

    if __model_enum == __MODEL_TYPES_ENUM.LOGISTIC_REGRESSION or __model_enum == __MODEL_TYPES_ENUM.GBT_CLASSIFIER:
        y_grades_df = pd.DataFrame(
            {'F': y_grades[0], 'D': y_grades[1], 'D+': y_grades[2], 'C-': y_grades[3], 'C': y_grades[4],
             'C+': y_grades[5], 'B-': y_grades[6], 'B': y_grades[7], 'B+': y_grades[8],
             'A-': y_grades[9], 'A': y_grades[10]})
        predictions = pd.concat([x_df, y_df, y_predict_df, y_grades_df], axis=1)
    else:
        predictions = pd.concat([x_df, y_df, y_predict_df], axis=1)

    predictions.to_csv(__results_folder / ('PREDICTION_' + postreq_name + '.csv'), index=False)

    return predictions['predicted score'].values, y_df[postreq_name].values, rr, acc, (rmse / 10), model


def stratify_and_split(filename):
    x_trains = []
    x_tests = []
    y_trains = []
    y_tests = []
    id_tests = []

    x, y, ids = get_prereq_table(filename)
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

            id_tests.extend(ids[test_index])
            x_trains.append(x_train)
            x_tests.append(x_test)
            y_trains.append(y_train)
            y_tests.append(y_test)

            (pd.concat(
                [pd.DataFrame(x_train, columns=x_columns),
                 pd.DataFrame(y_train, columns=[filename[:-4]])],
                axis=1)).to_csv(__folds_folder / (filename[:-4] + '_' +
                                                  __TRAIN_PREFIX + str(loop_count + 1) + '.csv'), encoding='utf-8',
                                index=False)

            (pd.concat(
                [pd.DataFrame(x_test, columns=x_columns),
                 pd.DataFrame(y_test, columns=[filename[:-4]])],
                axis=1)).to_csv(__folds_folder / (filename[:-4] + '_' +
                                                  __TEST_PREFIX + str(loop_count + 1) + '.csv'), encoding='utf-8',
                                index=False)
            loop_count += 1

    return x_trains, x_tests, y_trains, y_tests, x_columns, len(x), id_tests


def read_predict_write():
    if not (__model_enum == __MODEL_TYPES_ENUM.BAYESIAN_NETWORK and (__tree_type == __TREE_TYPES_ENUM.ALL or __tree_type == __TREE_TYPES_ENUM.IMMEDIATE)):
        print('Training and testing beginning. A counter will print after the completion of each training set. \n')
        if not os.path.exists(__folds_folder):
            os.makedirs(__folds_folder)
        if not os.path.exists(__results_folder):
            os.makedirs(__results_folder)

        big_predicted = []
        big_actual = []
        big_ids = []

        results_each_postreq = [[], [], [], [], []]

        counter = 0
        for filename in sorted(os.listdir(__data_folder)):
            filename = str(filename[:-4] + '.csv')
            x_train, x_test, y_train, y_test, x_columns, n_samples, ids = stratify_and_split(filename)
            if n_samples > __MIN_SAMPLES_FOR_PREDICTING:
                predicted, actual, rr, acc, nrmse, model = predict(filename[:-4], x_train, x_test, y_train, y_test,
                                                                   x_columns)

                big_predicted += list(predicted)
                big_actual += list(actual)
                big_ids += list(ids)
                results_each_postreq[0].append(filename[:-4])
                results_each_postreq[1].append(rr)
                results_each_postreq[2].append(acc)
                results_each_postreq[3].append(nrmse)
                results_each_postreq[4].append(n_samples)
                print(counter)
                counter += 1

        studentIds = pd.DataFrame(big_ids, columns=['student_id'])
        predictions = pd.DataFrame(big_predicted, columns=['predicted'])
        actuals = pd.DataFrame(big_actual, columns=['actual'])
        all_results = pd.concat([studentIds, predictions, actuals], axis=1)
        all_results.to_csv(__results_folder / ('ALL_COURSES_PREDICTIONS_' + __tree_type.name + "_" + __model_enum.name
                                               + '.csv'), index=False)

        all_stats = pd.DataFrame(
            {'postreq': results_each_postreq[0], 'r^2': results_each_postreq[1], 'accuracy': results_each_postreq[2],
             'nrmse': results_each_postreq[3], 'n': results_each_postreq[4]})
        all_stats.to_csv(__results_folder / ('ALL_COURSES_STATS_' + __tree_type.name + "_" + __model_enum.name + '.csv'),
                         index=False)

        print('Model training, testing, and evaluation completed. Files saved to: \'' + str(__results_folder) + '\' \n')


def save_models():
    if not (__model_enum == __MODEL_TYPES_ENUM.BAYESIAN_NETWORK and (
            __tree_type == __TREE_TYPES_ENUM.ALL or __tree_type == __TREE_TYPES_ENUM.IMMEDIATE)):
        print('Model saving beginning. \n')
        start_time = time.time()
        if not os.path.exists(__model_output):
            os.makedirs(__model_output)

        with parallel_backend('loky', n_jobs=-1):
            Parallel()(delayed(dump_model)(filename) for filename in os.listdir(__tuning_results_folder))

        print('Model saving completed in ' + str(round(time.time() - start_time, 2)) + 's. Files saved to: '
              + str(__model_output) + '\n')


def dump_model(filename):
    filename = str(filename[:-4] + '.csv')
    x, y, _ = get_prereq_table(filename)
    x_columns = list(x.columns.values)
    x = x.fillna(-1).values
    y = y.fillna(-1).values

    if not os.path.exists(__tuning_results_folder / (filename[:-4] + '.npy')):
        read_dictionary = None
    else:
        read_dictionary = np.load(__tuning_results_folder / (filename[:-4] + '.npy'), allow_pickle=True).item()

    if __model_enum == __MODEL_TYPES_ENUM.LOGISTIC_REGRESSION:
        if read_dictionary is None:
            model = LogisticRegression(random_state=__RANDOM_SEED)
        else:
            model = LogisticRegression(random_state=__RANDOM_SEED, **read_dictionary)
    elif __model_enum == __MODEL_TYPES_ENUM.GBT_CLASSIFIER:
        if read_dictionary is None:
            model = GradientBoostingClassifier(random_state=__RANDOM_SEED)
        else:
            model = GradientBoostingClassifier(random_state=__RANDOM_SEED, **read_dictionary)
    elif __model_enum == __MODEL_TYPES_ENUM.NU_SVR:
        if read_dictionary is None:
            model = NuSVR()
        else:
            model = NuSVR(**read_dictionary)
    elif __model_enum == __MODEL_TYPES_ENUM.GBT_REGRESSOR:
        model = GradientBoostingRegressor(random_state=__RANDOM_SEED)
    elif __model_enum == __MODEL_TYPES_ENUM.RANDOM_FOREST_REGRESSOR:
        if read_dictionary is None:
            model = RandomForestRegressor(random_state=__RANDOM_SEED)
        else:
            model = RandomForestRegressor(**read_dictionary, random_state=__RANDOM_SEED)

    model.fit(x, y)

    pickle.dump(model, open(__model_output / (filename[:-4] + '.pkl'), 'wb'))


def model_selection_string():
    string = "Enter one of the following for model type: \n"
    for m_type in __MODEL_TYPES_ENUM:
        string += " '" + str(m_type.value) + "': " + m_type.name + " \n"
    return string


if __name__ == "__main__":

    tune_or_predict = int(input("Enter one of the following process types: \n"
                                "'1': Tune hyperparameters \n"
                                "'2': Run predictions \n"
                                "'3': Save models \n"
                                "'4': Run All Predictions \n"
                                "'5': Tuning batch \n"))

    if tune_or_predict != 1 and tune_or_predict != 2 and tune_or_predict != 3 and tune_or_predict != 4\
            and tune_or_predict != 5:
        raise ValueError('An invalid process type was passed.')

    if tune_or_predict == 4:
        for tree_type in __TREE_TYPES_ENUM:
            for model_type in __MODEL_TYPES_ENUM:
                __model_enum = model_type
                __tree_type = tree_type
                __data_folder, __folds_folder, __results_folder, __tuning_results_folder, __model_output = set_paths()
                print(str(__tree_type.name) + ":" + str(__model_enum.name))
                read_predict_write()

    elif tune_or_predict == 5:
        for tree_type in __TREE_TYPES_ENUM:
            for model_type in __MODEL_TYPES_ENUM:
                if not (model_type == __MODEL_TYPES_ENUM.MEAN_ZEROR or model_type == __MODEL_TYPES_ENUM.MOD_ZEROR):
                    __model_enum = model_type
                    __tree_type = tree_type
                    __data_folder, __folds_folder, __results_folder, __tuning_results_folder, __model_output = set_paths()
                    print(str(__tree_type.name) + ":" + str(__model_enum.name))
                    hyperparameter_tuning()
    else:
        __tree_type = int(input("Enter one of the following for prereq type: \n '1': Root prerequisites \n"
                                " '2': Immediate prerequisites \n '3': All prerequisites \n"))

        if __tree_type not in __TREE_TYPES_ENUM._value2member_map_:
            raise ValueError('An invalid tree type was passed.')
        else:
            __tree_type = __TREE_TYPES_ENUM(__tree_type)

        __model_enum = int(input(model_selection_string()))
        print(__model_enum)
        if __model_enum not in __MODEL_TYPES_ENUM._value2member_map_:
            raise ValueError('An invalid model type was passed.')
        else:
            __model_enum = __MODEL_TYPES_ENUM(__model_enum)

        __data_folder, __folds_folder, __results_folder, __tuning_results_folder, __model_output = set_paths()

        if tune_or_predict == 1:
            hyperparameter_tuning()
        elif tune_or_predict == 2:
            read_predict_write()
        elif tune_or_predict == 3:
            save_models()
