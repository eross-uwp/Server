# Using Linear Regression to predict next term GPA

from sklearn.linear_model import LogisticRegression


def lr_predict(x, y):

    return LogisticRegression.predict(x, y)

