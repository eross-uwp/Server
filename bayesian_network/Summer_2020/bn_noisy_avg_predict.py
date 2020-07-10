"""
__Author__: Nick Tiede

__Purpose__:
"""
from pomegranate import *
import seaborn

from Summer_2020.con_prob_table_creator import get_disc_dist_list
from Summer_2020.csv_read_write import *
from Summer_2020.disc_dist_creator import create_real_state_list
from Summer_2020.noisy_avg_calc import create_target_cpt

seaborn.set_style('whitegrid') # Used by pomegranate

DATA_FILE = 'data\\oops2data.csv'

# Standard: 11, Whole letter grade: 5, Binary: 2
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

# Gets list of states (nodes) of prereqs that each contain a discrete distribution\
prereq_state_list = create_real_state_list(df_data, num_prereqs, NUM_GRADES)

# Creates the target course CPT pandas DataFrame based on the noisy-avg method
# This does not return a pomegranate State or ConditionalProbabilityTable so this DataFrame can be saved easier later
target_df_cpt = create_target_cpt(df_data, NUM_GRADES)

# Converts the target course CPT DataFrame to pomegranate objects
target_pom_cpt = ConditionalProbabilityTable(target_df_cpt.values.tolist(), get_disc_dist_list(prereq_state_list))
target_state = State(target_pom_cpt)
print("Created target course state \n")

# Adds all the course states (nodes) to the model
for state in prereq_state_list:
    model.add_state(state)

model.add_state(target_state)
print("All course states added to the model \n")

# Add edges pointing from the prereqs to the target course to create the structure of the Bayesian Network
for state in prereq_state_list:
    model.add_edge(state, target_state)
print("All edges added to the model \n")

model.bake()
print(model.structure)
print("Bayesian Network structure finalized \n")

# 0  1  2  3  4  5  6  7  8  9  10
# F  D  D+ C- C  C+ B- B  B+ A- A
print(model.predict([['7', '7', '7', None]]))
print("")
print(model.predict_proba([['7', '7', '7', None]]))
print("\n\n")

print(model.predict([['10', '9', '8', None]]))
print("")
print(model.predict_proba([['10', '9', '8', None]]))
print("\n\n")

print(model.predict([['7', '6', '8', None]]))
print("")
print(model.predict_proba([['7', '6', '8', None]]))
print("\n\n")

print(model.predict([['8', '5', '3', None]]))
print("")
print(model.predict_proba([['8', '5', '3', None]]))

print(model.predict([['5', '3', '0', None]]))
print("")
print(model.predict_proba([['5', '3', '0', None]]))
print("\n\n")

print(model.predict([['10', '10', '10', None]]))
print("")
print(model.predict_proba([['10', '10', '10', None]]))
print("\n\n")

print(model.predict([['0', '0', '0', None]]))
print("")
print(model.predict_proba([['0', '0', '0', None]]))
print("\n\n")
