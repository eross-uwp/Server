"""
___authors___: Chris Kott
"""

import os
import sys

import pandas as pd
import numpy as np
import enum
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.svm import NuSVR
import warnings
import time
from pathlib import Path
from joblib import Parallel, delayed, parallel_backend
import pickle
from scipy.stats import loguniform
from sklearn.utils import column_or_1d

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

prereq_type = "IMMEDIATE"
__MODEL_TYPES_ENUM = enum.IntEnum('__MODEL_TYPES_ENUM', 'LOGISTIC_REGRESSION GBT_CLASSIFIER NU_SVR GBT_REGRESSOR '
                                                        'RANDOM_FOREST_REGRESSOR MOD_ZEROR MEAN_ZEROR')
__RANDOM_SEED = np.int64(313131)
data_folder = Path('data/' + prereq_type + 'PrereqTables')
tuning_path = Path('TuningResults_only_prereqs/' + prereq_type)
experiment_name = 'CT-CIS'
experiment_folder = Path(experiment_name)
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
    x = file.drop(columns=[course_name, 'student_id', 'cumulative_gpa', 'prev_term_gpa', 'struggle', 'term_difference'])
    return x, y


def convert_grade(string_grade):
    if str(string_grade) == 'nan':
        return -1
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
            else:
                missing_data = pd.read_csv(Path('data/studentGradesPerCourse.csv'))
                current_data = current_data.drop(columns=col)
                current_data = current_data.merge(missing_data[['student_id', col]], on='student_id', how='inner')
                current_data[col] = current_data[col].apply(convert_grade)
    print(current_data)
    return current_data


def train_and_predict(model_type, course_name, training_dat, prediction_dat):
    if model_type == __MODEL_TYPES_ENUM.MOD_ZEROR or model_type == __MODEL_TYPES_ENUM.MEAN_ZEROR:
        if model_type == __MODEL_TYPES_ENUM.MOD_ZEROR:
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

    x, y = trim(training_dat, course_name)
    x = x.fillna(-1)
    y = y.fillna(-1)
    model.fit(x, y)
    predicts = model.predict(prediction_dat.drop(columns='student_id'))
    return [round_school(num) for num in predicts]


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    training_data = pd.read_csv(experiment_folder/training_file)
    for model_types in __MODEL_TYPES_ENUM:

        testing_students = pd.read_csv(experiment_folder/testing_file)
        # print(training_data)
        for course in training_data.columns:
            print(course)
            course_data = pd.read_csv(data_folder/(course + ".csv"))
            data = course_data[course_data.student_id.isin(training_data[course])]
            prediction_data = get_prediction_data()
            predictions = train_and_predict(model_types, course, data, prediction_data)
            print(predictions)
            testing_students[course] = predictions
        testing_students.to_csv(experiment_folder/(model_types.name + '_predictions.csv'), index=False)
