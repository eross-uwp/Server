import os
import pandas as pd
import numpy as np
import random
from sklearn import metrics

PREREQ_PROCESS_TYPE = 'All'
MODEL_PREFIX = 'ModZeroR_'
STRATIFIED_DATA_PATH = '..\\data\\' + PREREQ_PROCESS_TYPE + 'PrereqFolds\\'
TABLES_FILE_PATH = '..\\data\\' + PREREQ_PROCESS_TYPE + 'PrereqTables\\'
RESULTS_FOLDER = '..\\results\\' + PREREQ_PROCESS_TYPE + 'PrereqModZeroR\\'
random.seed = 313131
population = [0, 1]
possible_grades = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]


# Grabs the training and testing data the given course name and fold number.
def get_training_testing(course_name, number_of_fold):
    return pd.read_csv(STRATIFIED_DATA_PATH + course_name + '_train_' + str(number_of_fold) + '.csv'), pd.read_csv(
        STRATIFIED_DATA_PATH + course_name + '_test_' + str(number_of_fold) + '.csv')


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
    results_each_postreq = [[], [], [], [], []]

    prediction_array = None
    for postreq_name in os.listdir(TABLES_FILE_PATH):
        postreq_name = postreq_name[:-4]  # Get the name of the courses from the .npy files and drop the extension.

        try:
            train, test = get_training_testing(postreq_name, 1)
        except Exception as e:
            continue

        prediction_array = np.zeros(0)
        test_total = None
        for number_for_fold in range(1, 6):

            try:
                train, test = get_training_testing(postreq_name, number_for_fold)
                if set == 1:
                    test_total = test
                else:
                    test_total = pd.concat([test_total, test], axis=0, ignore_index=True)
                y_train = train[postreq_name].values
                y_test = test[postreq_name].values

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
                print(e)
                break

        if not os.path.exists(RESULTS_FOLDER):
            os.makedirs(RESULTS_FOLDER)

        # Calculate metrics.
        rr = metrics.r2_score(test_total[postreq_name].values, prediction_array)
        rmse = np.math.sqrt(metrics.mean_squared_error(test_total[postreq_name].values, prediction_array))
        acc = metrics.accuracy_score(test_total[postreq_name].values, prediction_array)
        # Save metrics.
        with open(RESULTS_FOLDER + MODEL_PREFIX + postreq_name + '.txt', "w") as text_file:
            text_file.write(
                'R^2 = ' + str(rr) + ', Accuracy = ' + str(acc) + ' , RMSE = ' + str(rmse) + ', NRMSE = ' + str(
                    rmse / 10))

        results_each_postreq[0].append(postreq_name)
        results_each_postreq[1].append(rr)
        results_each_postreq[2].append(acc)
        results_each_postreq[3].append(rmse / 10)
        results_each_postreq[4].append(y_train.size + y_test.size)

        # Make predictions a single vertical array and add it to pandas.
        predictions = pd.DataFrame.from_records(prediction_array.reshape(-1, 1), columns=['predicted'])
        # Concatenate test_total onto predictions.
        predictions = pd.concat([test_total, predictions], axis=1)
        # Write to csv file.
        predictions.to_csv(RESULTS_FOLDER + MODEL_PREFIX + 'PREDICTION_' + str(postreq_name) + '.csv', index=False)
        # Add to the big_predictions list.
        actual_data_frame = pd.DataFrame(test_total[postreq_name].values, columns=['actual'])
        big_predictions = pd.concat([big_predictions, pd.concat(
            [predictions['predicted'], pd.DataFrame(test_total[postreq_name].values, columns=['actual'])], axis=1)],
                                    axis=0)

    # Convert total metrics to data-frame.
    all_stats = pd.DataFrame(
        {'postreq': results_each_postreq[0], 'r^2': results_each_postreq[1], 'accuracy': results_each_postreq[2],
         'nrmse': results_each_postreq[3], 'n': results_each_postreq[4]})
    # Print total metrics.
    all_stats.to_csv(RESULTS_FOLDER + MODEL_PREFIX + 'ALL_COURSES_STATS.csv', index=False)

    # Compile list of all predictions, and write to file.
    all_predictions = np.empty((0, 2))
    for actual in big_predictions.itertuples():
        all_predictions = np.vstack(
            (all_predictions, [reverse_convert_grade(actual.predicted), reverse_convert_grade(actual.actual)]))
    pd.DataFrame(all_predictions, columns=['predicted', 'actual']).to_csv(
        RESULTS_FOLDER + MODEL_PREFIX + 'ALL_COURSES_PREDICTIONS' + '.csv', index=False)
    print('Done')
