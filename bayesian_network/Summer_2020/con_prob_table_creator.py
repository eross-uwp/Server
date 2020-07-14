"""
__Author__: Nick Tiede

__Purpose__: To create a pomegranate conditional probability table, whose probabilities will be overwritten
             from training, for the target course for prediction
"""
import pandas as pd
from itertools import product
from pomegranate import ConditionalProbabilityTable
from Summer_2020.cartesian_table_creator import create_cartesian_table


# Takes in the number of prereqs and the number of grades
# Creates a list of rows with every possible event and an equal probability for each
def create_con_prob_table(num_prereqs, num_grades, states):

    # Creates the cartesian product of the grades as a DataFrame
    df_events = create_cartesian_table(num_grades, num_prereqs + 1)

    # Adds a column of probabilities as floats to the DataFrame
    df_events[len(df_events.columns)] = 1/num_grades

    return ConditionalProbabilityTable(df_events.values.tolist(), get_disc_dist_list(states))


# Returns a list of discrete distributions from a list of states
def get_disc_dist_list(states):
    return [state.distribution for state in states]
