# Used for tuning the Gradient Boosted Trees classifier for nth Semester Graduation Prediction. Runs a GridSearchCV on
# a bunch of Gradient Boosted Trees classifier hyperparameters for each term.

"""
___authors___: Austin FitzGerald
"""

from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import StratifyAndGenerateDatasets as sd
from sklearn.model_selection import GridSearchCV, train_test_split

GRADUATED_HEADER = 'graduated'
FIRST_TERM = [sd.RAW_DATA_FIRST, sd.FIRST_HEADERS]
SECOND_TERM = [sd.RAW_DATA_SECOND, sd.SECOND_HEADERS]
THIRD_TERM = [sd.RAW_DATA_THIRD, sd.THIRD_HEADERS]
FOURTH_TERM = [sd.RAW_DATA_FOURTH, sd.FOURTH_HEADERS]
FIFTH_TERM = [sd.RAW_DATA_FIFTH, sd.FIFTH_HEADERS]
SIXTH_TERM = [sd.RAW_DATA_SIXTH, sd.SIXTH_HEADERS]
SEVENTH_TERM = [sd.RAW_DATA_SEVENTH, sd.SEVENTH_HEADERS]
EIGHTH_TERM = [sd.RAW_DATA_EIGHTH, sd.EIGHTH_HEADERS]
NINTH_TERM = [sd.RAW_DATA_NINTH, sd.NINTH_HEADERS]
TENTH_TERM = [sd.RAW_DATA_TENTH, sd.TENTH_HEADERS]
TERMS_ARRAY = [FIRST_TERM, SEVENTH_TERM, THIRD_TERM, FOURTH_TERM, FIFTH_TERM, SIXTH_TERM, SEVENTH_TERM, EIGHTH_TERM,
               NINTH_TERM, TENTH_TERM]
TUNING_RESULTS_FILE_PREFIX = 'GBTTuningResults\\term_'

if __name__ == "__main__":
    counter = 1
    for TERM_INDEX in TERMS_ARRAY:
        gradient = GradientBoostingClassifier(random_state=sd.RANDOM_SEED)

        parameters = {
            "loss": ["deviance"],
            "learning_rate": [0.01, 0.1, 0.5, 1],
            "min_samples_split": np.linspace(0.1, 0.5, 5),
            "min_samples_leaf": np.linspace(0.1, 0.5, 5),
            "max_depth": [1, 3, 4],
            "max_features": ["log2", "sqrt", 0.1, 0.5],
            "criterion": ["friedman_mse", "mae"],
            "subsample": [0.5, 0.8, 1.0],
            "n_estimators": [300, 500]
        }

        clf = GridSearchCV(gradient, parameters, cv=5, n_jobs=-1)

        term = pd.read_csv(TERM_INDEX[0])

        for col_name in term.columns:
            if term[col_name].dtype == 'object':
                term[col_name] = term[col_name].astype('category')
                term[col_name] = term[col_name].cat.codes

        X = term[TERM_INDEX[1]].copy().values

        y = term[GRADUATED_HEADER].copy().values.reshape(-1, 1).ravel()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=sd.RANDOM_SEED)
        best_clf = clf.fit(X_train, y_train)

        with open(TUNING_RESULTS_FILE_PREFIX + str(counter) + '.txt', 'w') as f:
            f.write(str(best_clf.best_params_))
        print('term #' + str(counter) + ' done.')

        counter += 1
