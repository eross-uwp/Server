import os
import sys

import pandas as pd
import numpy as np
from pip._internal.utils.misc import enum
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
import warnings


__data_folder = 'data\\AllPrereqTables\\'
__folds_folder = 'data\\Testing\\'
__model_enum = 0
__tree_type = 0
__tuning_results_folder = 'TuningResults\\All\\GBT\\'

__TRAIN_PREFIX = 'train_'
__TEST_PREFIX = 'test_'
__NUMBER_FOLDS = 5
__RANDOM_SEED = 313131
__MIN_SAMPLES_FOR_PREDICTING = 25

if __name__ == "__main__":
    while True:
        os.system('notepad.exe')
