# Using Linear Regression to predict next term GPA

"""
___authors___: Evan Majerus & Austin FitzGerald
"""

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

RANDOM_SEED = 313131

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


def lr_predict():
    model = LinearRegression()
    scores = []

    for i in range(0, 5):
        model.fit(X_train[i], y_train[i])
        score = model.score(X_test[i], y_test[i])
        scores.append(score)

    print(scores)


if __name__ == "__main__":
    lr_predict()
