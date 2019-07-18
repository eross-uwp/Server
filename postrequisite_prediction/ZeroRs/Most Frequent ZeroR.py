import os

import numpy as np
np.random.seed(0)
import pandas as pd
from random import choices
import random
import sys
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, accuracy_score

GRAPH_FILE_PREFIX = 'graph_term_'
PREREQ_PROCESS_TYPE = 'Immediate'
STRATIFIED_DATA_PATH = '..\\data\\' + PREREQ_PROCESS_TYPE + 'PrereqFolds\\'
TABLES_FILE_PATH = '..\\data\\' + PREREQ_PROCESS_TYPE + 'PrereqTables\\'
RESULTS_FOLDER = '..\\results\\' + PREREQ_PROCESS_TYPE + 'PrereqModZeroR\\'
TUNING_FILE_PATH = '..\\TuningResults\\' + PREREQ_PROCESS_TYPE + '\\LR\\'
random.seed = 313131
population = [0, 1]
possible_grades = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

flatten = lambda l: [item for sublist in l for item in sublist]


def get_training_testing(course_name, number_of_fold):
    return pd.read_csv(STRATIFIED_DATA_PATH + course_name + '_train_' + str(number_of_fold) + '.csv'),\
           pd.read_csv(STRATIFIED_DATA_PATH + course_name + '_test_' + str(number_of_fold) + '.csv')


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
    big_predictions = []

    prediction_array = None
    for each_course in os.listdir(TUNING_FILE_PATH):
        each_course = each_course[:-4]
        try:
            train, test = get_training_testing(each_course, 1)
        except Exception as e:
            continue
        prediction_array = np.zeros(0)
        test_total = None
        for number_for_fold in range(1, 6):
            try:
                train, test = get_training_testing(each_course, number_for_fold)
                y_train = train[each_course].values
                y_test = test[each_course].values
                grades_distribution = dict.fromkeys(possible_grades)

                for key, value in grades_distribution.items():
                    grades_distribution[key] = 0
                for each_grade in y_train:
                    grades_distribution[each_grade] = grades_distribution[each_grade] + 1
                prediction = max(grades_distribution, key=grades_distribution.get)

                for_conca = np.full_like(y_test, prediction)
                prediction_array = np.concatenate((prediction_array, for_conca), axis=0)
            except Exception as e:
                break

        for set in range(1, 6):
            train, test = get_training_testing(each_course, set)
            if set == 1:
                test_total = test
            else:
                test_total = pd.concat([test_total, test], axis=0, ignore_index=True)

        predictions = pd.DataFrame.from_records(prediction_array.reshape(-1,1), columns=['prediction'])
        predictions = pd.concat([test_total, predictions], axis=1)
        predictions.to_csv(RESULTS_FOLDER + str(each_course) + '.csv', index=False)
        big_predictions += list(predictions['prediction'].values)
    converted_grades = []
    for grade in big_predictions:
        converted_grades.append(reverse_convert_grade(grade))
    all_predictions = pd.DataFrame(converted_grades, columns=['predictions'])
    all_predictions.to_csv(RESULTS_FOLDER + 'ALL_PREDICTIONS' + '.csv', index=False)
