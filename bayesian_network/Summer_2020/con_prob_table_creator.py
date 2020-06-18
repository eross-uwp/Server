"""
__Author__: Nick Tiede

__Purpose__: To create a pomegranate conditional probability table, whose probabilities will be overwritten
             from training, for the target course for prediction
"""
import pandas as pd
import numpy as np
from itertools import product
from pomegranate import ConditionalProbabilityTable


# Takes in the number of prereqs and the number of grades
# Creates a list of rows with every possible event and an equal probability for each
def create_con_prob_table(num_prereqs, num_grades, states):

    # Creates the cartesian product of the grades, converts the list of touples to a list of lists,
    # then to a DataFrame, then converts all of the data to ints to get rid of decimals, then to strings
    df_events = pd.DataFrame(map(list, product(range(0, num_grades), repeat=num_prereqs + 1))).applymap(int).applymap(str)

    # Adds a column of probabilities as floats to the DataFrame
    df_events[len(df_events.columns)] = 1/num_grades

    return ConditionalProbabilityTable(df_events.values.tolist(), get_disc_dist_list(states))


# Returns a list of discrete distributions from a list of states
def get_disc_dist_list(states):
    return [state.distribution for state in states]
