"""
___authors___: Chris Kott
"""
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
from bayesian_network.Summer_2020 import bn_interface

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

__data_folder = Path('data/BayesNetTables')
__folds_folder = Path('data/BayesNetFolds')
__results_folder = Path('results/BayesNet')
__tuning_results_folder = Path('TuningResults/BayesNetCPTs')


__NUMBER_FOLDS = 5
__RANDOM_SEED = np.int64(313131)
__MIN_SAMPLES_FOR_PREDICTING = 25

np.random.seed(__RANDOM_SEED)




