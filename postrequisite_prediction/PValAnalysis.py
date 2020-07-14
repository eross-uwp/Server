"""
___authors___: Chris Kott
"""

import os
import sys

import pandas as pd
import numpy as np
import enum
import warnings
import time
from sklearn.utils import resample
from pathlib import Path
from numpy.random import RandomState
from joblib import Parallel, delayed
from itertools import combinations

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
                                                        'RANDOM_FOREST_REGRESSOR MOD_ZEROR MEAN_ZEROR')
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
    results_folder2 = Path('results/' + __tree_type.name + 'Prereq_' + __model_enumB.name + '_Results/')
    if not (os.path.exists(results_folder1) and os.path.exists(results_folder2)):
        raise FileNotFoundError("One or both of the file paths don't exist.")
    return results_folder1, results_folder2


def get_pval():
    __results_folder_A, __results_folder_B = set_paths()
    results_A = pd.read_csv(__results_folder_A / ('ALL_COURSES_PREDICTIONS_' + __tree_type.name + "_"
                                                  + __model_enumA.name + '.csv'))
    results_B = pd.read_csv(__results_folder_B / ('ALL_COURSES_PREDICTIONS_' + __tree_type.name + "_"
                                                  + __model_enumB.name + '.csv'))
    results_A['actual'] = results_A['actual'].apply(convert_grade)
    results_A['predicted'] = results_A['predicted'].apply(convert_grade)
    results_A['diff^2'] = (results_A['actual'] - results_A['predicted']) ** 2

    results_B['actual'] = results_B['actual'].apply(convert_grade)
    results_B['predicted'] = results_B['predicted'].apply(convert_grade)
    results_B['diff^2'] = (results_B['actual'] - results_B['predicted']) ** 2

    nrmses_a = np.sqrt(results_A.groupby('student_id').sum()['diff^2'] / results_A.groupby('student_id').size()) / 10
    nrmses_a = pd.DataFrame({'student_id': nrmses_a.index, __model_enumA.name: nrmses_a.values})

    nrmses_b = np.sqrt(results_B.groupby('student_id').sum()['diff^2'] / results_B.groupby('student_id').size()) / 10
    nrmses_b = pd.DataFrame({'student_id': nrmses_b.index, __model_enumB.name: nrmses_b.values})

    total = pd.merge(nrmses_a, nrmses_b, "inner", "student_id")
    if total[__model_enumA.name].mean() <= total[__model_enumB.name].mean():
        better_model = __model_enumA.name
        worse_model = __model_enumB.name
    else:
        better_model = __model_enumB.name
        worse_model = __model_enumA.name
    total['diff'] = total[better_model] - total[worse_model]
    start_time = time.time()
    bootstrapped_values = Parallel(n_jobs=-1)(delayed(resample)(total['diff'].values, random_state=__RANDOM_SEED + i)
                                              for i in range(250000))
    bootstrap = pd.DataFrame(bootstrapped_values)
    bootstrap = bootstrap.T

    bootstrap_means = bootstrap.mean().to_frame(name='mean_diff')
    p_val = bootstrap_means[bootstrap_means['mean_diff'] > 0].count().values[0] / bootstrap_means['mean_diff'].count()
    return better_model, worse_model, p_val


if __name__ == "__main__":
    for tree_type in __TREE_TYPES_ENUM:
        p_values = []
        print(tree_type.name)
        for val in combinations(list(map(int, __MODEL_TYPES_ENUM)), 2):
            __tree_type = tree_type
            __model_enumA = __MODEL_TYPES_ENUM(val[0])
            __model_enumB = __MODEL_TYPES_ENUM(val[1])
            p_val = get_pval()
            p_values.append(p_val)
            print(p_val)
        pd.DataFrame(p_values, columns=['better', 'worse', 'p_value'])\
            .to_csv(Path('results/') / (tree_type.name + '_PValues.csv'), index=False)
