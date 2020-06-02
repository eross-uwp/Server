import os
import pandas as pd
import numpy as np
import random

# from random import choices
# import sys
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, accuracy_score

np.random.seed(313131)
GRAPH_FILE_PREFIX = 'graph_term_'
PREREQ_PROCESS_TYPE = 'All'
STRATIFIED_DATA_PATH = '..\\data\\' + PREREQ_PROCESS_TYPE + 'PrereqFolds\\'
TABLES_FILE_PATH = '..\\data\\' + PREREQ_PROCESS_TYPE + 'PrereqTables\\'
RESULTS_FOLDER = '..\\results\\' + PREREQ_PROCESS_TYPE + 'PrereqModZeroR\\'
TUNING_FILE_PATH = '..\\TuningResults\\' + PREREQ_PROCESS_TYPE + '\\LR\\'
random.seed = 313131
population = [0, 1]
possible_grades = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]


# flatten = lambda l: [item for sublist in l for item in sublist]  # Unused?


# Grabs the training and testing data the given course name and fold number.
def get_training_testing(course_name, number_of_fold):
    return pd.read_csv(STRATIFIED_DATA_PATH + course_name + '_train_' + str(number_of_fold) + '.csv'), \
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
    big_predictions = None

    prediction_array = None
    for each_course in os.listdir(TUNING_FILE_PATH):
        each_course = each_course[:-4]  # Get the name of the courses from the .npy files and drop the extension.

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

                # Grab all possible grades and populate the grades_distribution dictionary with zeros.
                grades_distribution = dict.fromkeys(possible_grades)
                for key, value in grades_distribution.items():
                    grades_distribution[key] = 0

                # Increment each grade for each entry
                for each_grade in y_train:
                    grades_distribution[each_grade] = grades_distribution[each_grade] + 1

                # Find the grade with the most entries.
                prediction = max(grades_distribution, key=grades_distribution.get)

                # Create an array with the shape of y_test, but prediction in all of its values.
                for_concat = np.full_like(y_test, prediction)

                # Concatenate shaped prediction array onto prediction_array.
                prediction_array = np.concatenate((prediction_array, for_concat), axis=0)

            except Exception as e:
                break

        for set in range(1, 6):
            train, test = get_training_testing(each_course, set)
            if set == 1:
                test_total = test
            else:
                test_total = pd.concat([test_total, test], axis=0, ignore_index=True)

        # Make predictions a single vertical array and add it to pandas.
        predictions = pd.DataFrame.from_records(prediction_array.reshape(-1, 1), columns=['predicted'])
        # Concatenate test_total onto predictions.
        predictions = pd.concat([test_total, predictions], axis=1)
        # Write to csv file.
        predictions.to_csv(RESULTS_FOLDER + str(each_course) + '.csv', index=False)
        # Add to the big_predictions list.
        actual_data_frame = pd.DataFrame(test_total[each_course].values, columns=['actual'])
        big_predictions = pd.concat([big_predictions, pd.concat([predictions['predicted'],
                                                                 pd.DataFrame(test_total[each_course].values,
                                                                              columns=['actual'])], axis=1)], axis=0)

    # Compile list of all predictions, and write to file.
    all_predictions = np.empty((0, 2))
    for actual in big_predictions.itertuples():
        all_predictions = np.vstack(
            (all_predictions, [reverse_convert_grade(actual.predicted), reverse_convert_grade(actual.actual)]))
    pd.DataFrame(all_predictions, columns=['predicted', 'actual']).to_csv(RESULTS_FOLDER + 'ALL_PREDICTIONS' + '.csv',
                                                                          index=False)
    print('Done')
