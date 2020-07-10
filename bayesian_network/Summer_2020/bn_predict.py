"""
__Author__: Nick Tiede

__Purpose__: To create a standard Bayesian network model to predict from based on input data and the number of grades.
             bn_interface should be used to call this function.
"""
from pomegranate import *
import seaborn
from Summer_2020.csv_read_write import *
from Summer_2020.disc_dist_creator import create_disc_dist_state_list
from Summer_2020.con_prob_table_creator import create_con_prob_table


# Creates an entire standard Bayesian network and returns the model to be predicted from
def create_std_bn(data_file, num_grades=11):
    seaborn.set_style('whitegrid')  # Used by pomegranate for some reason

    # Initializes the model
    model = BayesianNetwork()
    print("Standard Bayesian Network initialized \n")

    # Reads in the data as a pandas DataFrame
    df_data = read_data_csv(data_file)
    # print(df_data)
    print("Reading data complete \n")

    # Gets the number of prereqs for the course
    num_prereqs = len(df_data.columns) - 1
    print("Prereqs: " + str(num_prereqs) + "\n")

    # Creats a list of states (nodes) for each prereq that includes a course name and a discrete distribution
    disc_dist_states_list = create_disc_dist_state_list(df_data.columns, num_prereqs, num_grades)
    print(disc_dist_states_list)
    print("Created discrete distributions and discrete states \n")

    # Creates the state (node) for the course that is meant to be predicted
    target_course_state = \
        State(create_con_prob_table(num_prereqs, num_grades, disc_dist_states_list), df_data.columns[num_prereqs])
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
    print("Bayesian Network structure finalized: " + model.structure + "\n")

    # Converts the data to a usable format for pomegranate then trains the model with the data
    df_data_float = df_data.astype(float)
    df_data_float.replace({float('nan'): np.nan}, regex=True)
    model = BayesianNetwork.from_structure(df_data, structure=model.structure)
    print("Bayesian Network trained \n")

    return model
