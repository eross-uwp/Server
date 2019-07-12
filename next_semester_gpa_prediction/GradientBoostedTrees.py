"""
___authors___: Nate Braukhoff and Zhiwei Yang and Austin FitzGerald
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor

import BaseDataSetGenerator as bD

RESULTS_FOLDER = 'GBTRegressorResults\\'
GRAPH_FILE_PREFIX = 'graphs\\graph_'
RESULTS_TEXTFILE = 'GBTRegressor_Results.txt'
PREDICTION_OUTPUT_PREFIX = 'predictions'
__x_train = []
__y_train = []
__ids_test = []
__prev_terms_test = []
__next_terms_test = []
__x_test = []
__y_test = []


def get_training_testing():
    # Creating arrays that contain arrays holding the testing and training data. Reshaped to form a 1 row multi
    # column array

    for i in range(0, bD.NUMBER_OF_FOLDS):
        __x_train.append(
            pd.read_csv('data\\test_train\\train_' + str(i + 1) + '.csv')['prev GPA'].values.reshape(-1, 1))
        __y_train.append(
            pd.read_csv('data\\test_train\\train_' + str(i + 1) + '.csv')['next GPA'].values.reshape(-1, 1))
        __ids_test.append(pd.read_csv('data\\test_train\\test_' + str(i + 1) + '.csv')['id'].values.reshape(-1, 1))
        __prev_terms_test.append(
            pd.read_csv('data\\test_train\\test_' + str(i + 1) + '.csv')['prev term number'].values.reshape(-1, 1))
        __next_terms_test.append(
            pd.read_csv('data\\test_train\\test_' + str(i + 1) + '.csv')['next term number'].values.reshape(-1, 1))
        __x_test.append(pd.read_csv('data\\test_train\\test_' + str(i + 1) + '.csv')['prev GPA'].values.reshape(-1, 1))
        __y_test.append(pd.read_csv('data\\test_train\\test_' + str(i + 1) + '.csv')['next GPA'].values.reshape(-1, 1))


def gbt_predict():
    np.random.seed(bD.RANDOM_SEED)
    model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, max_features=1, max_depth=4, loss='ls',
                                      random_state=bD.RANDOM_SEED)

    y_preds = []

    for i in range(0, bD.NUMBER_OF_FOLDS):
        # fitting the model and storing the predicted value from the test set
        model.fit(__x_train[i], __y_train[i])
        y_pred = model.predict(__x_test[i])

        y_preds += list(y_pred)  # the predicted next term gpa

        for j in range(0, len(__x_test[i])):
            plt.plot([__x_test[i][j], __x_test[i][j]], [__y_test[i][j], y_pred[j]], color='k',
                     zorder=1)  # the lines between real and predicted

        plt.scatter(__x_test[i], __y_test[i], color='g', label='real',
                    zorder=2)  # the real data from the tests, in green
        plt.scatter(__x_test[i], y_pred, color='r', label='predicted',
                    zorder=2)  # the predicted data from the tests, in red

        plt.title('test #' + str(i + 1))
        plt.xlabel('Prev term GPA')
        plt.ylabel('Next term GPA')
        plt.legend(loc='upper left')
        plt.savefig(RESULTS_FOLDER + GRAPH_FILE_PREFIX + str(i + 1))  # saving graphs
        plt.close()

        # Calculating the stats from the actual curr-term GPAs and predicted curr-term GPAs
    rr = metrics.r2_score(flatten(__y_test), y_preds)
    rmse = np.math.sqrt(metrics.mean_squared_error(flatten(__y_test), y_preds)) / 4

    # Saving the stats to a text file.
    with open(RESULTS_FOLDER + RESULTS_TEXTFILE, "w") as text_file:
        text_file.write('R^2 = ' + str(rr) + ', RMSE = ' + str(rmse))

    # save predictions (matching with tests) to files
    predictions = pd.DataFrame(
        {'id': flatten(__ids_test), 'prev term': flatten(__prev_terms_test), 'next term': flatten(__next_terms_test),
         'prev term gpa': flatten(__x_test), 'next term gpa': flatten(__y_test), 'next term gpa prediction': y_preds})
    predictions.to_csv(RESULTS_FOLDER + PREDICTION_OUTPUT_PREFIX + '.csv', index=False)


flatten = lambda l: [item for sublist in l for item in
                     sublist]  # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists

if __name__ == "__main__":
    get_training_testing()
    gbt_predict()
