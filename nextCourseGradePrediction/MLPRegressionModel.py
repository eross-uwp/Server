# Use a MLP regressor to predict the next course grade given the previous course, grade, as well as the next course name

"""
___authors___: Austin FitzGerald
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# CONSTANTS
RAW_DATA_FILE = 'data\\raw_data.csv'
FINAL_DATA_FILE = 'data\\finalDataSet.csv'
TESTING_TRAINING_DATA_FOLDER = 'data\\test_train\\'
TRAIN_PREFIX = 'train_'
TEST_PREFIX = 'test_'
FIRST_COLUMN = 'id'
SECOND_COLUMN = 'term'
THIRD_COLUMN = 'prev course'
FOURTH_COLUMN = 'prev grade'
FIFTH_COLUMN = 'next course'
SIXTH_COLUMN = 'next grade'
RAW_DATA_FRAME_HEADERS = [FIRST_COLUMN, SECOND_COLUMN, THIRD_COLUMN, FOURTH_COLUMN, FIFTH_COLUMN, SIXTH_COLUMN]
RANDOM_SEED = 313131
NUMBER_OF_FOLDS = 5

np.random.seed(RANDOM_SEED)

if __name__ == "__main__":
    raw_data = pd.read_csv(RAW_DATA_FILE, names=RAW_DATA_FRAME_HEADERS)  # our raw dataset
    X = raw_data.drop(['next grade', 'id', 'term'], axis=1)
    y = raw_data['next grade']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    y_tests = []
    y_preds = []
    mlp = MLPRegressor(hidden_layer_sizes=(25, 25, 25), max_iter=5500, random_state=RANDOM_SEED)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    y_tests += list(y_test)  # the real value
    y_preds += list(y_pred)  # the predicted value
    rr = mlp.score(X_test, y_test)
    rmse = np.math.sqrt(metrics.mean_squared_error(y_tests, y_preds))
    plt.scatter(X_test['prev grade'], y_test, color='g', label='real')
    plt.scatter(X_test['prev grade'], mlp.predict(X_test), color='r', label='predicted')
    plt.plot(X_test['prev grade'], mlp.predict(X_test), color='k', label='predicted')  # the linear regression line
    plt.xlabel('Curr course grade')
    plt.ylabel('Next course grade')
    plt.legend(loc='upper left')
    plt.show()
    print("R^2: " + str(rr) + ", RMSE: " + str(rmse))
