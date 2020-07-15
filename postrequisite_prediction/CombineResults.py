import enum
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

__MODEL_TYPES_ENUM = enum.IntEnum('__MODEL_TYPES_ENUM', 'GBT_CLASSIFIER LOGISTIC_REGRESSION MOD_ZEROR MEAN_ZEROR '
                                                        'GBT_REGRESSOR NU_SVR RANDOM_FOREST_REGRESSOR')
__TREE_TYPES_ENUM = enum.IntEnum('__TREE_TYPES_ENUM', 'ROOT IMMEDIATE ALL')

for tree_type in __TREE_TYPES_ENUM:
    dataframe = pd.DataFrame()
    for model_type in __MODEL_TYPES_ENUM:
        results_folder = Path('results/' + tree_type.name + 'Prereq_' + model_type.name + '_Results/')
        file = pd.read_csv(results_folder / ('ALL_COURSES_PREDICTIONS_' + tree_type.name + "_" + model_type.name +
                                             '.csv'))
        if model_type == 1:
            dataframe['actual'] = file['actual']
        dataframe[model_type.name] = file['predicted']
    dataframe.to_csv(Path('results/')/(tree_type.name+'_Combined_Predictions.csv'), index=False)
