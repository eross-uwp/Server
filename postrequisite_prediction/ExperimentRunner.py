"""
___authors___: Chris Kott
"""

import enum
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import random
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import NuSVR
from bayesian_network.Summer_2020 import bn_interface

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

prereq_type = "ALL"
__MODEL_TYPES_ENUM = enum.IntEnum('__MODEL_TYPES_ENUM', 'LOGISTIC_REGRESSION GBT_CLASSIFIER NU_SVR GBT_REGRESSOR '
                                                        'RANDOM_FOREST_REGRESSOR MOD_ZEROR MEAN_ZEROR BAYESIAN_NETWORK')
__RANDOM_SEED = np.int64(313131)
random.seed(__RANDOM_SEED)
data_folder = Path('data/' + prereq_type + 'PrereqTables')
tuning_path = Path('TuningResults/' + prereq_type)
experiment_list = ['CT-CIS', 'SE-CIS', 'SE-CT[30-59]', 'SE-CT[60-89]']  #
experiment_name = 'SE-CIS'
experiment_folder = Path('Experiments/' + experiment_name)
testing_file = 'TESTING_STUDENTS.csv'
training_file = 'TRAINING_STUDENTS.csv'


# https://stackoverflow.com/a/43886290
def round_school(x):
    if x < 0:
        return 0
    if x > 10:
        return 10
    else:
        i, f = divmod(x, 1)
        return int(i + ((f >= 0.5) if (x > 0) else (f > 0.5)))


def trim(file, course_name):
    y = file[course_name]
    if model_types == __MODEL_TYPES_ENUM.BAYESIAN_NETWORK:
        x = file.drop(columns=[course_name, 'student_id'])
    else:
        x = file.drop(columns=[course_name, 'student_id', 'cumulative_gpa', 'prev_term_gpa', 'struggle', 'term_difference'])
    return x, y


def convert_grade(string_grade):
    if str(string_grade) == 'nan':
        return string_grade
    grade_conversions = {
        'A': 10,
        'A-': 9,
        'B+': 8,
        'B': 7,
        'B-': 6,
        'C+': 5,
        'C': 4,
        'C-': 3,
        'D+': 2,
        'D': 1,
        'F': 0
    }
    return grade_conversions[string_grade]


def get_prediction_data():
    current_data = course_data[course_data.student_id.isin(testing_students[experiment_name])]
    current_data = current_data[0:0]
    if model_types == __MODEL_TYPES_ENUM.BAYESIAN_NETWORK:
        current_data = current_data.drop(columns=[course])
    else:
        current_data = current_data.drop(columns=[course, 'cumulative_gpa', 'prev_term_gpa', 'struggle', 'term_difference'])
    for col in current_data.columns:
        if col == 'student_id':
            current_data[col] = testing_students[experiment_name]
        else:
            if col in testing_students.columns:
                testData = testing_students[[experiment_name, col]]
                testData.columns = ['student_id', col]
                current_data = current_data.drop(columns=col)
                current_data = current_data.merge(testData, on='student_id', how='inner')
            elif col == "Basic Algebra":
                current_data[col] = current_data[col].apply(convert_grade)
                continue
            else:
                missing_data = pd.read_csv(Path('data/studentGradesPerCourse.csv'), index_col=False)
                current_data = current_data.drop(columns=col)
                current_data = current_data.merge(missing_data[['student_id', col]], on='student_id', how='inner')
                current_data[col] = current_data[col].apply(convert_grade)
    return current_data


def train_and_predict(model_type, course_name, training_dat, prediction_dat):
    print(model_type.name, ":", course_name)
    x, y = trim(training_dat, course_name)
    if model_type == __MODEL_TYPES_ENUM.BAYESIAN_NETWORK:
        da = pd.concat([x, y], axis=1).astype(pd.Int64Dtype()).astype(str).replace({'<NA>': 'nan'})
        model = bn_interface.create_bayesian_network(da)
    elif model_type == __MODEL_TYPES_ENUM.MOD_ZEROR:
        model = DummyClassifier('most_frequent')
    elif model_type == __MODEL_TYPES_ENUM.MEAN_ZEROR:
        model = DummyRegressor('mean')
    else:
        read_dictionary = np.load(tuning_path/model_type.name/(course_name + ".npy"), allow_pickle=True).item()
        if model_type == __MODEL_TYPES_ENUM.LOGISTIC_REGRESSION:
            model = LogisticRegression(random_state=__RANDOM_SEED, **read_dictionary)
        elif model_type == __MODEL_TYPES_ENUM.GBT_CLASSIFIER:
            model = GradientBoostingClassifier(random_state=__RANDOM_SEED, **read_dictionary)
        elif model_type == __MODEL_TYPES_ENUM.NU_SVR:
            model = NuSVR(**read_dictionary)
        elif model_type == __MODEL_TYPES_ENUM.GBT_REGRESSOR:
            model = GradientBoostingRegressor(random_state=__RANDOM_SEED)
        elif model_type == __MODEL_TYPES_ENUM.RANDOM_FOREST_REGRESSOR:
            model = RandomForestRegressor(**read_dictionary, random_state=__RANDOM_SEED)
        else:
            raise NotImplementedError('Invalid Model Type')

    if model_type != __MODEL_TYPES_ENUM.BAYESIAN_NETWORK:
        x = x.fillna(-1)
        y = y.fillna(-1)
        model.fit(x, y)
        prediction_dat = prediction_dat.fillna(-1)
        predicts = model.predict(prediction_dat.drop(columns='student_id'))
    else:
        dat = prediction_dat.drop(columns='student_id').astype(pd.Int64Dtype()).astype(str).replace({'<NA>': 'nan'}).values
        predicts = bn_interface.bn_multi_predict(model, dat)
        predicts = [int(i) for i in predicts]
    return [round_school(num) for num in predicts]


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    for experiment in experiment_list:
        print(experiment)
        experiment_name = experiment
        experiment_folder = Path('Experiments/' + experiment_name)
        training_data = pd.read_csv(experiment_folder/training_file, index_col=False)
        for model_types in __MODEL_TYPES_ENUM:
            testing_students = pd.read_csv(experiment_folder/testing_file, index_col=False)
            # print(training_data)
            for course in training_data.columns:
                print(course)
                if model_types == __MODEL_TYPES_ENUM.BAYESIAN_NETWORK:
                    course_data = pd.read_csv(Path('data/BayesNetTables')/(course + ".csv"), index_col=False)
                else:
                    course_data = pd.read_csv(data_folder/(course + ".csv"), index_col=False)
                data = course_data[course_data.student_id.isin(training_data[course])]
                prediction_data = get_prediction_data()
                predictions = train_and_predict(model_types, course, data, prediction_data)
                print(predictions)
                testing_students[course] = predictions
            testing_students.to_csv(experiment_folder/(model_types.name + '_predictions.csv'), index=False)
