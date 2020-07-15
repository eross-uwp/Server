import math
import os
from pathlib import Path

import numpy as np

data_folder = Path('data/AllPrereqTables/')
tuning_results_folder_old = Path('TuningResults_original/All/GBT/')
tuning_results_folder_new = Path('TuningResults/All/GBT/')

for filename in os.listdir(tuning_results_folder_old):
    old_results = np.load(tuning_results_folder_old / (filename[:-4] + ".npy"), allow_pickle=True).item()
    new_results = np.load(tuning_results_folder_new / (filename[:-4] + ".npy"), allow_pickle=True).item()
    print(filename[:-4])

    if old_results != new_results:
        print(sorted(old_results.items()))
        print(sorted(new_results.items()))
    print()

