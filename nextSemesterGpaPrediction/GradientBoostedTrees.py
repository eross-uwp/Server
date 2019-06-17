"""
___authors___: Nate Braukhoff
Preparation of the data:
    1. Convert objects into number - done with data set
    2. Fill nulls with 0.0
    3. Use MinMaxScaler() to transform the data
    4. Randomly split training set into train and validation subsets


Output:
    Learning rate
    Training accuracy score
    Validation accuracy score
    Confusion Matrix
    Classification Report
"""
# import machine learning algorithms
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd


if __name__ == "__main__":
    # Read in Tables
    train = pd.read_csv('data\\GBT_Data\\Train.csv')
    test = pd.read_csv('data\\GBT_Data\\Test.csv')

    # print("Train")
    # print(train)
    # print("Test")
    # print(test)

    # setting x values
    train.set_index("prev GPA", inplace=True)
    test.set_index("prev GPA", inplace=True)

    # print(train.index)
    # print(test.index)

    # setting y values
    y_train = train["current GPA"]

    # print(y_train.values)

    # Drop current GPA from the train data set
    train.drop(labels="current GPA", axis=1, inplace=True)

    # print(train)

    train_test = train.append(test, sort=False)

    # print(train_test)

    dropped_columns = ["id", "prev term number", "current term number"]
    train_test.drop(labels=dropped_columns, axis=1, inplace=True)

    # print(train_test)

    train_test_dummies = pd.get_dummies(train_test, columns=["current GPA"])
    train_test_dummies.fillna(value=0.0, inplace=True)

    # print(train.shape)

    x_train = train_test_dummies.values[0:1376]
    x_test = train_test_dummies.values[1376:]

    # print(x_test)
    # print(x_train)

    # Transforming the data
    scaler = MinMaxScaler()
    x_train_scale = scaler.fit_transform(x_train)
    x_test_scale = scaler.transform(x_test)

    # Splitting the data
    x_train_sub, x_validation_sub, y_train_sub, y_validation_sub = train_test_split(x_train_scale, y_train,
                                                                                    random_state=0)

    learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
    for learning_rate in learning_rates:
        gb = GradientBoostingRegressor(n_estimators=20, learning_rate=learning_rate, max_features=1, max_depth=2)

        gb.fit(x_train_sub, y_train_sub)
        print("Learning rate: ", learning_rate)
        print("Training accuracy score: {0:.3f}".format(gb.score(x_train_sub, y_train_sub)))
        print("Validation accuracy score: {0:.3f}".format(gb.score(x_validation_sub, y_validation_sub)))
        print()

    gb = GradientBoostingRegressor(n_estimators=2, learning_rate = 0.5, max_features=1, max_depth=2)
    gb.fit(x_train_sub, y_train_sub)

    print("Confusion Matrix")
    print(confusion_matrix(y_train_sub, x_train_sub))

