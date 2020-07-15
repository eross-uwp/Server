"""
__Author__: Nick Tiede

__Purpose__: To create a standard Bayesian network model to predict from based on input data and the number of grades.
             bn_interface should be used to call this function. This is an outdated file. bn_std_model should be used.
             This is due to pomegranate not being able to handle learning a Bayesian network from samples when
             some of the potential grade values are missing. For example, if a course never has an instance of a C,
             then pomegranate assumes that C is not a possibility for that course and does not include it. This causes
             errors when trying to predict given some of these missing values.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import seaborn
from bayesian_network.Summer_2020.con_prob_table_creator import create_con_prob_table
from bayesian_network.Summer_2020.bayecsv_read_write import *
from bayesian_network.Summer_2020.disc_dist_creator import create_disc_dist_state_list
from pomegranate import *


# Creates an entire standard Bayesian network and returns the model to be predicted from
def create_std_bn_old(df_data, num_grades=11):
    seaborn.set_style('whitegrid')  # Used by pomegranate for some reason

    # Initializes the model
    model = BayesianNetwork()
    print("Standard Bayesian Network initialized")

    # Gets the number of prereqs for the course
    num_prereqs = len(df_data.columns) - 1
    print("Prereqs: " + str(num_prereqs))

    # Creats a list of states (nodes) for each prereq that includes a course name and a discrete distribution
    disc_dist_states_list = create_disc_dist_state_list(df_data.columns, num_prereqs, num_grades)
    print(disc_dist_states_list)
    print("Created discrete distributions and discrete states")

    # Creates the state (node) for the course that is meant to be predicted
    target_course_state = \
        State(create_con_prob_table(num_prereqs, num_grades, disc_dist_states_list), df_data.columns[num_prereqs])
    print("Created target course state")

    # Adds all the course states (nodes) to the model
    for state in disc_dist_states_list:
        model.add_state(state)

    model.add_state(target_course_state)
    print("All course states added to the model")

    # Add edges pointing from the prereqs to the target course to create the structure of the Bayesian Network
    for state in disc_dist_states_list:
        model.add_edge(state, target_course_state)
    print("All edges added to the model")

    model.bake()
    print("Bayesian Network structure finalized: " + str(model.structure))

    # Converts the data to a usable format for pomegranate then trains the model with the data
    df_data_float = df_data.astype(float)
    df_data_float.replace({float('nan'): np.nan}, regex=True)
    model = BayesianNetwork.from_structure(df_data, structure=model.structure)
    print("Bayesian Network trained \n")

    return model
