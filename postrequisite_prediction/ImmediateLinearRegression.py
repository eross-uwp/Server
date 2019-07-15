"""
___authors___: Evan Majerus & Austin FitzGerald
"""
import os

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold

__DATA_FOLDER = 'data\\ImmediatePrereqTables\\'
__FOLDS_OUTPUT = 'data\\ImmediatePrereqFolds\\'
__RESULTS_FOLDER = 'results\\ImmediatePrereqResults\\'
__TRAIN_PREFIX = 'train_'
__TEST_PREFIX = 'test_'
__NUMBER_FOLDS = 5
__RANDOM_SEED = 313131
__MIN_SAMPLES_FOR_PREDICTING = 25
np.random.seed(__RANDOM_SEED)


def get_prereq_table(filename):
    file = pd.read_csv(__DATA_FOLDER + '\\' + filename)
    y = file.iloc[:, 1]
    x = file.drop(file.columns[1], axis=1)

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
    model = LinearRegression()
    y_preds = []

    for fold_num in range(0, __NUMBER_FOLDS):
        model.fit(x_train[fold_num], y_train[fold_num])
        y_pred = model.predict(x_test[fold_num])
        y_preds += list(y_pred)
    rr = metrics.r2_score(flatten(y_test), y_preds)
    rmse = np.math.sqrt(metrics.mean_squared_error(flatten(y_test), y_preds))

    with open(__RESULTS_FOLDER + postreq_name + '.txt', "w") as text_file:
        text_file.write('R^2 = ' + str(rr) + ', RMSE = ' + str(rmse))
    x_df = pd.concat([pd.DataFrame(x_test[0]),
                                                pd.DataFrame(x_test[1]),
                                                pd.DataFrame(x_test[2]),
                                                pd.DataFrame(x_test[3]),
                                                pd.DataFrame(x_test[4])], ignore_index=True)
    x_df.columns = x_columns
    y_df = pd.concat([pd.DataFrame(y_test[0]),
                                                pd.DataFrame(y_test[1]),
                                                pd.DataFrame(y_test[2]),
                                                pd.DataFrame(y_test[3]),
                                                pd.DataFrame(y_test[4])], ignore_index=True)
    y_df.columns = [postreq_name]
    y_predict_df = pd.DataFrame(y_preds, columns=['predicted score'])
    predictions = pd.concat([x_df, y_df, y_predict_df], axis=1)
    predictions.to_csv(__RESULTS_FOLDER + 'PREDICTION_' + filename + '.csv', index=False)


flatten = lambda l: [item for sublist in l for item in
                     sublist]  # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists


if __name__ == "__main__":
    for filename in os.listdir(__DATA_FOLDER):
        x_train, x_test, y_train, y_test, x_columns = stratify_and_split(filename)
        if len(x_train) > 0:
            lr_predict(filename[:-4], x_train, x_test, y_train, y_test, x_columns)