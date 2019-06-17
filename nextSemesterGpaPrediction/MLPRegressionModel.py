# Using a Multilayer Perceptron Regressor to predict next term GPA

"""
___authors___: Austin FitzGerald
"""
from sklearn import metrics
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import BaseDataSetGenerator as bd

RESULTS_FOLDER = 'MLPRegressionResults\\'
GRAPH_FILE_PREFIX = 'MLPRegression_graph_'
RESULTS_TEXTFILE = 'MLPRegression_Results.txt'
MLP_HIDDEN_LAYERS = 100
MLP_MAX_ITERATIONS = 1000

# Creating arrays that contain arrays holding the testing and training data. Reshaped to form a 1 row multi column array
X_train = np.array([pd.read_csv('data\\test_train\\train_1.csv')['prev GPA'].values.reshape(-1, 1),
                    pd.read_csv('data\\test_train\\train_2.csv')['prev GPA'].values.reshape(-1, 1),
                    pd.read_csv('data\\test_train\\train_3.csv')['prev GPA'].values.reshape(-1, 1),
                    pd.read_csv('data\\test_train\\train_4.csv')['prev GPA'].values.reshape(-1, 1),
                    pd.read_csv('data\\test_train\\train_5.csv')['prev GPA'].values.reshape(-1, 1)])

y_train = np.array([pd.read_csv('data\\test_train\\train_1.csv')['current GPA'].values.reshape(-1, 1),
                    pd.read_csv('data\\test_train\\train_2.csv')['current GPA'].values.reshape(-1, 1),
                    pd.read_csv('data\\test_train\\train_3.csv')['current GPA'].values.reshape(-1, 1),
                    pd.read_csv('data\\test_train\\train_4.csv')['current GPA'].values.reshape(-1, 1),
                    pd.read_csv('data\\test_train\\train_5.csv')['current GPA'].values.reshape(-1, 1)])

X_test = np.array([pd.read_csv('data\\test_train\\test_1.csv')['prev GPA'].values.reshape(-1, 1),
                   pd.read_csv('data\\test_train\\test_2.csv')['prev GPA'].values.reshape(-1, 1),
                   pd.read_csv('data\\test_train\\test_3.csv')['prev GPA'].values.reshape(-1, 1),
                   pd.read_csv('data\\test_train\\test_4.csv')['prev GPA'].values.reshape(-1, 1),
                   pd.read_csv('data\\test_train\\test_5.csv')['prev GPA'].values.reshape(-1, 1)])

y_test = np.array([pd.read_csv('data\\test_train\\test_1.csv')['current GPA'].values.reshape(-1, 1),
                   pd.read_csv('data\\test_train\\test_2.csv')['current GPA'].values.reshape(-1, 1),
                   pd.read_csv('data\\test_train\\test_3.csv')['current GPA'].values.reshape(-1, 1),
                   pd.read_csv('data\\test_train\\test_4.csv')['current GPA'].values.reshape(-1, 1),
                   pd.read_csv('data\\test_train\\test_5.csv')['current GPA'].values.reshape(-1, 1)])


def mlp():
    np.random.seed(bd.RANDOM_SEED)
    model = MLPRegressor(hidden_layer_sizes=(MLP_HIDDEN_LAYERS, MLP_HIDDEN_LAYERS), max_iter=MLP_MAX_ITERATIONS)

    # hold all tests and predictions in order to calculate R^2 AND RMSE.
    y_tests = []
    y_preds = []

    for i in range(0, 5):
        model.fit(X_train[i], y_train[i])
        y_pred = model.predict(X_test[i])

        y_tests += list(y_test[i])  # the real value
        y_preds += list(y_pred)  # the predicted value

        plt.scatter(X_test[i], y_test[i], color='g', label='real')  # the real data from the tests, in green
        # plt.scatter(X_test[i], y_pred, color='r', label='predicted')  # the predicted data from the tests, in red
        plt.plot(X_test[i], model.predict(X_test[i]), color='k', label='predicted')  # the linear regression line
        plt.title('test #' + str(i + 1))
        plt.xlabel('Prev term GPA')
        plt.ylabel('Curr term GPA')
        plt.legend(loc='upper left')
        plt.savefig(RESULTS_FOLDER + GRAPH_FILE_PREFIX + str(i + 1))  # saving graphs
        plt.close()

    # Calculating the R^2 and RMSE from the actual curr-term GPAs and predicted curr-term GPAs
    rr = metrics.r2_score(y_tests, y_preds)
    rmse = np.math.sqrt(metrics.mean_squared_error(y_tests, y_preds)) / 4

    # Saving the R^2 and RMSE to a text file.
    with open(RESULTS_FOLDER + RESULTS_TEXTFILE, "w") as text_file:
        text_file.write('R^2 = ' + str(rr) + ', RMSE = ' + str(rmse))


if __name__ == "__main__":
    mlp()
