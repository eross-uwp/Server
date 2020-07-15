import enum
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from scipy.stats import loguniform
from sklearn import metrics
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import NuSVR
from sklearn.utils import column_or_1d


def remove_terms(val):
    new_val = str(val)
    if new_val == 'nan':
        return np.NaN
    return new_val.split(',')[1]


grade_list_path = Path('data/student_grade_list_with_terms.csv')
out_file_path = Path('data/studentGradesPerCourse.csv')

grade_list = pd.read_csv(grade_list_path)

if __name__ == "__main__":
    for col in grade_list.columns:
        if col != 'student_id':
            grade_list[col] = grade_list[col].apply(remove_terms)
    grade_list.to_csv(out_file_path, index=False)
