import numpy as np
np.random.seed(0)
import pandas as pd
from random import choices
import random
import sys
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, accuracy_score
import statistics

GRAPH_FILE_PREFIX = 'graph_term_'
STRATIFIED_DATA_PATH = '..\\data\\ImmediatePrereqFolds\\'
RESULTS_FOLDER = '..\\results\\ImmediatePrereqMeanZeroR\\'
UNIQUE_COURSE = '..\\data\\uniqueCourses.csv'
random.seed = 313131
population = [0, 1]
possible_grades = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

flatten = lambda l: [item for sublist in l for item in sublist]

def get_training_testing(course_name, number_of_fold):
    return pd.read_csv(STRATIFIED_DATA_PATH + course_name + '_train_' + str(number_of_fold) + '.csv'),\
           pd.read_csv(STRATIFIED_DATA_PATH + course_name + '_test_' + str(number_of_fold) + '.csv')

if __name__ == "__main__":

    unique_course = pd.read_csv(UNIQUE_COURSE)
    prediction_array = None
    for each_course in unique_course['unique_courses']:
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
                #prediction = max(grades_distribution, key=grades_distribution.get)
                sum = 0
                count = 0
                for grade, amount in grades_distribution.items():
                    sum = sum + grade*amount
                    count = count + amount
                prediction = sum / count


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
        predictions = pd.DataFrame.from_records(prediction_array.reshape(-1,1), columns=['prob of prediction'])
        predictions = pd.concat([test_total, predictions], axis=1)
        predictions.to_csv(RESULTS_FOLDER + str(each_course) + '.csv', index=False)
