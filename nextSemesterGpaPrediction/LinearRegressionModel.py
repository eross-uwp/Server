# Using Linear Regression to predict next term GPA

"""
___authors___: Evan Majerus & Austin FitzGerald
"""
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import BaseDataSetGenerator as bd

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
    np.random.seed(bd.RANDOM_SEED)
    model = LinearRegression()

    ytests = []
    ypreds = []

    for i in range(0, 5):
        # fitting the model and storing the predicted value from the test set
        model.fit(X_train[i], y_train[i])
        y_pred = model.predict(X_test[i])

        ytests += list(y_test[i])  # the real value
        ypreds += list(y_pred)  # the predicted value

        # graphing. x-axis is real prev term GPAs, y-axis is real current term GPAs, line is the linear regression
        plt.scatter(X_test[i], y_test[i], color='g', )
        plt.plot(X_test[i], model.predict(X_test[i]), color='k')
        plt.title('test #' + str(i+1))
        plt.show()

    # Calculating the R^2 and RMSE from the actual curr-term GPAs and predicted curr-term GPAs
    rr = metrics.r2_score(ytests, ypreds)
    rmse_error = np.math.sqrt(metrics.mean_squared_error(ytests, ypreds))
    print("R^2: {:.5f}%, RMSE: {:.5f}".format(rr * 100, rmse_error))


if __name__ == "__main__":
    lr_predict()
