"""
__Author__: Nick Tiede

__Purpose__: An interface to our noisy-avg Bayesian network and the standard Bayesian network.
             generate_navg_cpt() should be used to create and save CPTS then create_bayesian_network() to get model
"""
import copy
import math
from timeit import default_timer as timer
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import pandas as pd
import pathlib
import numpy as np
from bayesian_network.Summer_2020.bn_noisy_avg_model import create_navg_bn
from bayesian_network.Summer_2020.bn_std_model import create_std_bn
from bayesian_network.Summer_2020.noisy_avg_calc import create_target_cpt


# Combines noisy-avg cpt loading, generating, and saving functions
# Also makes the CPTs easily readable when saved
# Takes in full data file path and the folder file path to save CPT
def generate_navg_cpt(data_loc, save_loc, num_grades=11, reverse=False, verbose=True, check_dup=True):
    start_time = timer()

    df_data = load_data_csv(data_loc, reverse)
    if verbose: print('Generating ' + df_data.columns[-1] + ' CPT')
    file_name = df_data.columns[-1] + ' CPT'
    df_cpt = create_navg_cpt(df_data, num_grades=num_grades, check_dup=check_dup)
    if isinstance(save_loc, pathlib.Path):
        save_cpt_as_csv(df_cpt, save_loc / file_name)
    else:
        save_cpt_as_csv(df_cpt, save_loc + file_name)

    end_time = timer()
    final_time = end_time - start_time
    t_hr = str(math.trunc((final_time / 60) / 60))
    t_min = str(math.trunc(final_time / 60) - (int(t_hr) * 60))
    t_sec = str(round(final_time - (int(t_min) * 60), 2))
    if verbose: print('Created ' + file_name + ' in ' + t_hr + ' hrs ' + t_min + ' min ' + t_sec + ' sec \n')
    return


# Returns a DataFrame of a noisy-avg CPT given a DataFrame of grade data
def create_navg_cpt(df_data, num_grades=11, check_dup=True):
    col_names = list(df_data.columns.values)
    col_names.append('probability')
    df_cpt = create_target_cpt(df_data, num_grades, check_dup=check_dup)
    df_cpt.columns = col_names
    return df_cpt


# Creates and returns a bayesian network model to be predicted from
# Current valid model types are 'noisyavg' and 'standard'
# num_grades options - Standard: 11, whole letter grade: 5, binary: 2 (others also work)
# df_cpt should be used with loaded in CPT DataFrame
def create_bayesian_network(df_data, num_grades=11, model_type='noisyavg', df_cpt=None, verbose=False):
    if model_type == 'noisyavg':
        return create_navg_bn(df_data, num_grades=num_grades, df_cpt=df_cpt, verbose=verbose)
    elif model_type == 'standard':
        return create_std_bn(df_data, num_grades=num_grades, verbose=verbose)
    else:
        print('Bayesian network type not valid: Use model_type= standard or noisyavg as strings')
    return


# Wrapper function for pomegranate predict method
# prereq_grade_list should be a list of strings of decimalized grades
# Returns a single predicted decimalized target course grade as a string
def bn_predict(bn_model, prereq_grade_list):
    grade_list = copy.deepcopy(prereq_grade_list.copy())
    grade_list.append(None)
    return str(bn_model.predict([grade_list])[-1][-1])


# Wrapper function for pandas .to_csv()
# Saves Dataframe CPT as a CSV at a specified file location
def save_cpt_as_csv(dataframe, file_loc):
    dataframe.to_csv(file_loc, index=False)
    return


# Wrapper function for pandas .read_csv()
# Loads CSV CPT from file location to DataFrame
def load_cpt_from_csv(file_loc):
    df_cpt = pd.read_csv(file_loc)
    df_cpt = df_cpt.astype(str)
    df_cpt['probability'] = df_cpt['probability'].astype(float)
    return df_cpt


# Loads in data from csv and formats it if there are nan values
# reverse variable reverses order of columns if data has target course first
def load_data_csv(file_loc, reverse=False):
    df_data = pd.read_csv(file_loc)

    for col in df_data.columns:
        if df_data[col].dtype == np.float64:
            df_data[col] = df_data[col].astype(pd.Int64Dtype()).astype(str)
            df_data[col] = df_data[col].replace({'<NA>': 'nan'})

    df_data = df_data.astype(str)

    if reverse:
        return df_data.iloc[:, ::-1]
    else:
        return df_data
