from postrequisite_prediction.OrdinalClassifier import OrdinalClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from statistics import mean
import numpy as np
import os
import warnings
import sys
from joblib import parallel_backend

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses


def round_school(x):
    if x < 0:
        return 0
    if x > 10:
        return 10
    else:
        i, f = divmod(x, 1)
        return int(i + ((f >= 0.5) if (x > 0) else (f > 0.5)))


def rounding_rmse_scorer(y, y_pred):
    return -mean_squared_error(y, [round_school(num) for num in y_pred], squared=False)

with parallel_backend('loky', n_jobs=-1):
    for filename in os.listdir(Path('data/AllPrereqTables')):
        data_file = pd.read_csv(Path('data/AllPrereqTables/')/filename)

        y = data_file.iloc[:, 1]
        x = data_file.drop([data_file.columns[1], data_file.columns[0]], axis=1)  # drop the postreq grade and student_id columns

        x = x.fillna(-1)
        y = y.values
        x = x.values
        if len(y) > 25:

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=313131)
            '''
            x_trains = []
            x_tests = []
            y_trains = []
            y_tests = []
    
            for train_index, test_index in skf.split(x, y):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
    
                x_trains.append(x_train)
                x_tests.append(x_test)
                y_trains.append(y_train)
                y_tests.append(y_test)
            '''
            model = OrdinalClassifier(RandomForestClassifier(random_state=313131))

            num_trials = 1500
            param_grid = {
                "n_estimators": np.logspace(np.log10(10), np.log10(1500), 100, dtype='int64'),
                "criterion": ["friedman_mse", "mae", "mse"],
                "min_samples_split": list(range(1, len(y), 1)),
                "min_samples_leaf": list(range(1, len(y), 1)),
                "max_features": ["auto", "sqrt", "log2"]
            }
            scoring = make_scorer(rounding_rmse_scorer)
            clf = RandomizedSearchCV(model, param_grid, cv=skf, scoring=scoring, n_iter=num_trials,
                                     random_state=313131,
                                     verbose=True)
            best_clf = clf.fit(x, y)
            '''
            rmses = []
            for i in range(0, 5):
                model.fit(x_trains[i], y_trains[i])
                predictions = model.predict(x_tests[i])
                rmse = mean_squared_error(y_tests[i], predictions, squared=False)
                rmses.append(rmse)
                # print(predictions)
                # print(rmse)'''
            print(filename, ":", best_clf.best_score_)



