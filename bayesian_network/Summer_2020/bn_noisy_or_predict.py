"""
__Author__: Nick Tiede

__Purpose__:
"""
from pomegranate import *
import seaborn
from Summer_2020.csv_read_write import *
from Summer_2020.disc_dist_creator import create_disc_dist_state_list
from Summer_2020.con_prob_table_creator import create_con_prob_table
from Summer_2020.noisy_or_calc import get_probabilities

seaborn.set_style('whitegrid') # Used by pomegranate

DATA_FILE = 'data\\oops2data.csv'

# Standard: 11, whole letter grade: 5, binary: 2
NUM_GRADES = 11

# Initializes the model
model = BayesianNetwork()
print("Bayesian Network initialized \n")

# Reads in the data as a pandas DataFrame
df_data = read_data_csv(DATA_FILE)
print(df_data)
print("Reading data complete \n")

# Gets the number of prereqs for the course
num_prereqs = len(df_data.columns) - 1
print("Prereqs: " + str(num_prereqs) + "\n")

get_probabilities(df_data, NUM_GRADES)