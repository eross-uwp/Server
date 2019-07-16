"""
___authors___: Austin FitzGerald
"""
import os

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

__DATA_FOLDER = 'data\\AllPrereqTables\\'
__RESULTS_FOLDER = 'results\\AllPrereq_GBTClassifier_Results\\'
__TRAIN_PREFIX = 'train_'
__TEST_PREFIX = 'test_'
__NUMBER_FOLDS = 5
__RANDOM_SEED = 313131
__MIN_SAMPLES_FOR_PREDICTING = 25
np.random.seed(__RANDOM_SEED)


def get_prereq_table(filename):
    file = pd.read_csv(__DATA_FOLDER + '\\' + filename)
    y = file.iloc[:, 1]
    x = file.drop([file.columns[1], file.columns[0]], axis=1) # drop the postreq grade and student_id columns

    return x, y


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
        for train_index, test_index in skf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            x_trains.append(x_train)
            x_tests.append(x_test)
            y_trains.append(y_train)
            y_tests.append(y_test)

    return x_trains, x_tests, y_trains, y_tests, x_columns


def lr_predict(postreq_name, x_train, x_test, y_train, y_test, x_columns):
    model = GradientBoostingClassifier(random_state=__RANDOM_SEED)
    y_preds = []
            #   F,  D,  D+, C-, C,  C+, B-, B,  B+, A-, A
    y_grades = [[], [], [], [], [], [], [], [] ,[] ,[], []]

    for fold_num in range(0, __NUMBER_FOLDS):
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
    rmse = np.math.sqrt(metrics.mean_squared_error(flatten(y_test), y_preds))
    acc = metrics.accuracy_score(flatten(y_test), y_preds)

    with open(__RESULTS_FOLDER + postreq_name + '.txt', "w") as text_file:
        text_file.write('R^2 = ' + str(rr) +', Accuracy = ' + str(acc) + ' , RMSE = ' + str(rmse) + ', NRMSE = ' + str(rmse/10))
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

    y_grades_df = pd.DataFrame({'F':y_grades[0], 'D':y_grades[1], 'D+':y_grades[2], 'C-':y_grades[3], 'C':y_grades[4],
                                'C+':y_grades[5], 'B-':y_grades[6], 'B':y_grades[7], 'B+':y_grades[8],
                                'A-':y_grades[9], 'A':y_grades[10]})
    predictions = pd.concat([x_df, y_df, y_predict_df, y_grades_df], axis=1)
    predictions.to_csv(__RESULTS_FOLDER + 'PREDICTION_' + filename, index=False)


flatten = lambda l: [item for sublist in l for item in
                     sublist]  # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists


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
    for filename in os.listdir(__DATA_FOLDER):
        x_train, x_test, y_train, y_test, x_columns = stratify_and_split(filename)
        if len(x_train) > 0:
            lr_predict(filename[:-4], x_train, x_test, y_train, y_test, x_columns)