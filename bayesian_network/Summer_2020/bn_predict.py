"""
__Author__: Nick Tiede

__Purpose__:
"""
from pomegranate import *
import seaborn;
from Summer_2020.csv_read_write import *
from Summer_2020.disc_dist_creator import create_disc_dist_state_list
from Summer_2020.con_prob_table_creator import create_con_prob_table

seaborn.set_style('whitegrid') # Used by pomegranate

DATA_FILE = 'data\\oops2data.csv'

# Standard: 11, whole letter grade: 6, binary: 2
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

# Creats a list of states (nodes) for each prereq that includes a course name and a discrete distribution
disc_dist_states_list = create_disc_dist_state_list(df_data.columns, num_prereqs, NUM_GRADES)
print(disc_dist_states_list)
print("Created discrete distributions and discrete states \n")

# Creates the state (node) for the course that is meant to be predicted
target_course_state = \
    State(create_con_prob_table(num_prereqs, NUM_GRADES, disc_dist_states_list), df_data.columns[num_prereqs])
# print(target_course_state)
print("Created target course state \n")

# Adds all the course states (nodes) to the model
for state in disc_dist_states_list:
    model.add_state(state)

model.add_state(target_course_state)
print("All course states added to the model \n")

# Add edges pointing from the prereqs to the target course to create the structure of the Bayesian Network
for state in disc_dist_states_list:
    model.add_edge(state, target_course_state)
print("All edges added to the model \n")

model.bake()
print(model.structure)
print("Bayesian Network structure finalized \n")

# Fitting using real OOPS2 grades and prereq grades
print(df_data.values.tolist())
model.fit(df_data.values.tolist())
print("Data fit complete \n")

print(model.structure)
