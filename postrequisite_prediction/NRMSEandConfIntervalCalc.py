"""
___authors___: Chris Kott
"""

import enum
import os
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.random import RandomState
from sklearn.utils import resample

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

__data_folder = Path()
__folds_folder = Path()
__results_folder_A = Path()
__results_folder_B = Path()
__tuning_results_folder = Path()
__model_output = Path()

__RANDOM_SEED = 313131
__MODEL_TYPES_ENUM = enum.IntEnum('__MODEL_TYPES_ENUM', 'LOGISTIC_REGRESSION GBT_CLASSIFIER NU_SVR GBT_REGRESSOR '
                                                        'RANDOM_FOREST_REGRESSOR MOD_ZEROR MEAN_ZEROR BAYESIAN_NETWORK')
__TREE_TYPES_ENUM = enum.IntEnum('__TREE_TYPES_ENUM', 'ROOT IMMEDIATE ALL')

__model_enumA = 0
__model_enumB = 0
__tree_type = 0

np.random.seed(__RANDOM_SEED)


def convert_grade(string_grade):
    if string_grade == 'A':
        return 10
    elif string_grade == 'A-':
        return 9
    elif string_grade == 'B+':
        return 8
    elif string_grade == 'B':
        return 7
    elif string_grade == 'B-':
        return 6
    elif string_grade == 'C+':
        return 5
    elif string_grade == 'C':
        return 4
    elif string_grade == 'C-':
        return 3
    elif string_grade == 'D+':
        return 2
    elif string_grade == 'D':
        return 1
    elif string_grade == 'F':
        return 0


def set_paths():
    results_folder1 = Path('results/' + __tree_type.name + 'Prereq_' + __model_enumA.name + '_Results/')
    if __model_enumA == __MODEL_TYPES_ENUM.BAYESIAN_NETWORK:
        results_folder1 = Path('results/BayesNet')
    if not (os.path.exists(results_folder1)):
        raise FileNotFoundError("The file paths don't exist.")
    return results_folder1


def get_nrmse():
    __results_folder_A = set_paths()
    if __model_enumA == __MODEL_TYPES_ENUM.BAYESIAN_NETWORK:
        results_A = pd.read_csv(__results_folder_A / 'ALL_COURSES_PREDICTIONS_BayesNet.csv')
    else:
        results_A = pd.read_csv(__results_folder_A / ('ALL_COURSES_PREDICTIONS_' + __tree_type.name + "_"
                                                      + __model_enumA.name + '.csv'))
    results_A['actual'] = results_A['actual'].apply(convert_grade)
    results_A['predicted'] = results_A['predicted'].apply(convert_grade)
    results_A['diff^2'] = (results_A['actual'] - results_A['predicted']) ** 2

    nrmses_a = np.sqrt(results_A.groupby('student_id').sum()['diff^2'] / results_A.groupby('student_id').size()) / 10
    nrmses_a = pd.DataFrame({'student_id': nrmses_a.index, __model_enumA.name: nrmses_a.values})

    bootstrapped_values = Parallel(n_jobs=-1)(delayed(resample)(nrmses_a[__model_enumA.name].values, random_state=__RANDOM_SEED + i)
                                              for i in range(250000))
    bootstrap = pd.DataFrame(bootstrapped_values).T

    bootstrap_means = bootstrap.mean().to_frame(name='nrmses')
    sorted_bootstrap_means = sorted(bootstrap_means['nrmses'].values)

    return __model_enumA.name, nrmses_a[__model_enumA.name].mean(), sorted_bootstrap_means[6249], sorted_bootstrap_means[231249]


if __name__ == "__main__":
    for tree_type in __TREE_TYPES_ENUM:
        nrmses = []
        print(tree_type.name)
        for val in __MODEL_TYPES_ENUM:
            if not ((val == __MODEL_TYPES_ENUM.BAYESIAN_NETWORK) and ((tree_type == __TREE_TYPES_ENUM.ROOT) or (tree_type == __TREE_TYPES_ENUM.IMMEDIATE))):
                __tree_type = tree_type
                __model_enumA = val
                value = get_nrmse()
                nrmses.append(value)
                print(value)
        pd.DataFrame(nrmses, columns=['model', 'NRMSE', 'lower_bound', 'upper_bound']).to_csv(Path('results/') / (tree_type.name + '_NRMSEs.csv'), index=False)
