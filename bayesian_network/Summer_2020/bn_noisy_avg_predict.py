"""
__Author__: Nick Tiede

__Purpose__: To create our noisy-avg Bayesian network model to predict from based on input data and
             the number of grades. bn_interface should be used to call this function.
"""
from pomegranate import *
import seaborn

from Summer_2020.con_prob_table_creator import get_disc_dist_list
from Summer_2020.csv_read_write import *
from Summer_2020.disc_dist_creator import create_real_state_list
from Summer_2020.noisy_avg_calc import create_target_cpt


# Creates an entire noisy-avg Bayesian network and returns the model to be predicted from
def create_navg_bn(data_file, num_grades=11, df_cpt=None):
    seaborn.set_style('whitegrid')  # Used by pomegranate

    # Initializes the model
    model = BayesianNetwork()
    print("Noisy-Avg Bayesian Network initialized \n")

    # Reads in the data as a pandas DataFrame
    df_data = read_data_csv(data_file)
    # print(df_data)
    print("Reading data complete \n")

    # Gets the number of prereqs for the course
    num_prereqs = len(df_data.columns) - 1
    print("Prereqs: " + str(num_prereqs) + "\n")

    # Gets list of states (nodes) of prereqs that each contain a discrete distribution\
    prereq_state_list = create_real_state_list(df_data, num_prereqs, num_grades)

    # Creates the target course CPT pandas DataFrame based on the noisy-avg method
    # or uses one already created
    if df_cpt is None:
        target_df_cpt = create_target_cpt(df_data, num_grades)
    else:
        target_df_cpt = df_cpt

    print("Bayesian Network trained \n")

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
    print("Bayesian Network structure finalized: " + model.structure + "\n")

    return model
