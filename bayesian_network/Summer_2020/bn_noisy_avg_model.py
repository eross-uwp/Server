"""
__Author__: Nick Tiede

__Purpose__: To create our noisy-avg Bayesian network model to predict from based on input data and
             the number of grades. bn_interface should be used to call this function.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import seaborn
from bayesian_network.Summer_2020.con_prob_table_creator import get_disc_dist_list
from bayesian_network.Summer_2020.disc_dist_creator import create_real_state_list
from bayesian_network.Summer_2020.noisy_avg_calc import create_target_cpt
from pomegranate import *


# Creates an entire noisy-avg Bayesian network and returns the model to be predicted from
def create_navg_bn(df_data, num_grades=11, df_cpt=None, verbose=False):
    seaborn.set_style('whitegrid')  # Used by pomegranate

    # Initializes the model
    model = BayesianNetwork()
    if verbose: print("Noisy-Avg Bayesian Network initialized")

    # Gets the number of prereqs for the course
    num_prereqs = len(df_data.columns) - 1
    if verbose: print("Prereqs: " + str(num_prereqs))

    # Gets list of states (nodes) of prereqs that each contain a discrete distribution
    prereq_state_list = create_real_state_list(df_data, num_prereqs, num_grades)
    if verbose: print("Created prereq states")

    # Creates the target course CPT pandas DataFrame based on the noisy-avg method
    # or uses one already created
    if df_cpt is None:
        target_df_cpt = create_target_cpt(df_data, num_grades)
    else:
        target_df_cpt = df_cpt

    if verbose: print("Bayesian Network trained")

    # Converts the target course CPT DataFrame to pomegranate objects
    target_pom_cpt = ConditionalProbabilityTable(target_df_cpt.values.tolist(), get_disc_dist_list(prereq_state_list))
    target_state = State(target_pom_cpt)
    if verbose: print("Created target course state")

    # Adds all the course states (nodes) to the model
    for state in prereq_state_list:
        model.add_state(state)

    model.add_state(target_state)
    if verbose: print("All course states added to the model")

    # Add edges pointing from the prereqs to the target course to create the structure of the Bayesian Network
    for state in prereq_state_list:
        model.add_edge(state, target_state)
    if verbose: print("All edges added to the model")

    model.bake()
    if verbose: print("Bayesian Network structure finalized: " + str(model.structure) + "\n")

    return model
