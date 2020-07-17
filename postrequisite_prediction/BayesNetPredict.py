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
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
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


flatten = lambda l: [item for sublist in l for item in
                     sublist]  # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists


def get_prereq_table(filename):
    file = pd.read_csv(__data_folder / filename)
    y = file.iloc[:, 1]
    ids = file['student_id']
    x = file.drop([file.columns[1], file.columns[0]], axis=1)  # drop the postreq grade and student_id columns
    # x = x.drop(x.columns[len(x.columns) - 1], axis=1)  # drop the term diff column
    # x = x.drop([x.columns[len(x.columns) - 1], x.columns[len(x.columns) - 2], x.columns[len(x.columns) - 3],
    #            x.columns[len(x.columns) - 4]], axis=1)  # remove all but prereqs.
    return x, y, ids


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


if __name__ == "__main__":
    if not os.path.exists(__tuning_results_folder):
        os.makedirs(__tuning_results_folder)
    if not os.path.exists(__results_folder):
        os.makedirs(__results_folder)

    big_predicts = []
    big_actuals = []
    big_ids = []

    big_courses = []
    big_r2 = []
    big_accuracy = []
    big_nrmse = []
    big_n = []
    for course in sorted(os.listdir(__data_folder)):
        print(course)
        y_tests = []
        x_tests = []
        id_tests = []
        total_predicts = []

        x, y, ids = get_prereq_table(course)
        x_columns = list(x.columns.values)
        x = x.values
        y = y.values
        if len(x) >= __MIN_SAMPLES_FOR_PREDICTING and len(y) >= __MIN_SAMPLES_FOR_PREDICTING:
            skf = StratifiedKFold(n_splits=__NUMBER_FOLDS, shuffle=True, random_state=__RANDOM_SEED)
            loop_count = 1
            for train_index, test_index in skf.split(x, y):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                test_ids = ids[test_index]

                da = pd.concat([pd.DataFrame(x_train, columns=x_columns), pd.DataFrame(y_train, columns=[course[:-4]])], axis=1).astype(pd.Int64Dtype())\
                    .astype(str).replace({'<NA>': 'nan'})
                if os.path.exists(__tuning_results_folder/(course[:-4] + "_CPT_" + str(loop_count) + ".csv")):
                    df_cpt = bn_interface.load_cpt_from_csv(__tuning_results_folder/(course[:-4] + "_CPT_" + str(loop_count) + ".csv"))
                else:
                    df_cpt = bn_interface.create_navg_cpt(da)
                    df_cpt.to_csv(__tuning_results_folder/(course[:-4] + "_CPT_" + str(loop_count) + ".csv")
                                  , index=False)
                model = bn_interface.create_bayesian_network(da, df_cpt=df_cpt)

                test_data = pd.DataFrame(x_test, columns=x_columns).astype(pd.Int64Dtype()).astype(str).replace({'<NA>': 'nan'}).values
                predicts = bn_interface.bn_multi_predict(model, test_data)
                predicts = [int(i) for i in predicts]

                total_predicts += list(predicts)
                id_tests += list(test_ids)
                x_tests.append(x_test)
                y_tests.append(y_test)

                print(loop_count)

                loop_count += 1

            rr = metrics.r2_score(flatten(y_tests), total_predicts)
            rmse = metrics.mean_squared_error(flatten(y_tests), total_predicts, squared=False)
            acc = metrics.accuracy_score(flatten(y_tests), total_predicts)

            big_r2.append(rr)
            big_nrmse.append(rmse/10)
            big_accuracy.append(acc)
            big_courses.append(course[:-4])
            big_n.append(len(total_predicts))

            with open(__results_folder / (course[:-4] + '.txt'), "w") as text_file:
                text_file.write(
                    'R^2 = ' + str(rr) + ', Accuracy = ' + str(acc) + ' , RMSE = ' + str(rmse) + ', NRMSE = ' + str(
                        rmse / 10))

            x_df = pd.concat([pd.DataFrame(x_tests[0]),
                              pd.DataFrame(x_tests[1]),
                              pd.DataFrame(x_tests[2]),
                              pd.DataFrame(x_tests[3]),
                              pd.DataFrame(x_tests[4])], ignore_index=True)

            x_df.columns = x_columns

            y_df = pd.concat([pd.DataFrame(y_tests[0]),
                              pd.DataFrame(y_tests[1]),
                              pd.DataFrame(y_tests[2]),
                              pd.DataFrame(y_tests[3]),
                              pd.DataFrame(y_tests[4])], ignore_index=True)

            y_df.columns = [course[:-4]]
            y_df[course[:-4]] = y_df[course[:-4]].apply(reverse_convert_grade)

            y_predict_df = pd.DataFrame(total_predicts, columns=['predicted score'])
            y_predict_df['predicted score'] = y_predict_df['predicted score'].apply(reverse_convert_grade)

            predictions = pd.concat([x_df, y_df, y_predict_df], axis=1)

            big_ids += list(id_tests)
            big_actuals += list(predictions[course[:-4]].values)
            big_predicts += list(predictions['predicted score'].values)

            predictions.to_csv(__results_folder / ('PREDICTION_' + course[:-4] + '.csv'), index=False)

    studentIds = pd.DataFrame(big_ids, columns=['student_id'])
    predictions = pd.DataFrame(big_predicts, columns=['predicted'])
    actuals = pd.DataFrame(big_actuals, columns=['actual'])
    all_results = pd.concat([studentIds, predictions, actuals], axis=1)
    all_results.to_csv(__results_folder / 'ALL_COURSES_PREDICTIONS_BayesNet.csv', index=False)

    courses = pd.DataFrame(big_courses, columns=['postreq'])
    rr = pd.DataFrame(big_r2, columns=['r^2'])
    accuracy = pd.DataFrame(big_accuracy, columns=['accuracy'])
    nrmse = pd.DataFrame(big_nrmse, columns=['nrmse'])
    n = pd.DataFrame(big_n, columns=['n'])
    all_stats = pd.concat([courses, rr, accuracy, nrmse, n], axis=1)
    all_stats.to_csv(__results_folder / ('ALL_COURSES_STATS_' + 'BayesNet.csv'), index=False)
