"""
__Author__: Nick Tiede

__Purpose__: To create a pomegranate state (node) list, whose probabilities will be overwritten from training,
             for each of the prereqs or to create a true state list from data that won't be overwritten
"""
import pandas as pd
import numpy as np
from pomegranate.base import State
from pomegranate.distributions.DiscreteDistribution import DiscreteDistribution


# Returns a of list of generated states (nodes) for each prereq
def create_disc_dist_state_list(course_names, num_prereqs, num_grades):
    state_list = []

    for i in range(0, num_prereqs):
        disc_dist = create_disc_dist(num_grades)
        state_list.append(State(disc_dist, course_names[i]))

    return state_list


# Returns a of list of real data states (nodes) for each prereq
def create_real_state_list(df_course_data, num_prereqs, num_grades):
    state_list = []
    df_structure = create_disc_dist_structure(num_grades)

    # For each prereq, make a DataFrame out of the individual course data, count the number of times each grade
    # occurs, turn that into probabilities for each grade, then convert to use with pomegranate
    for i in range(0, num_prereqs):
        df_prereq = pd.DataFrame({df_course_data.columns[i]: df_course_data.iloc[:, i]})
        df_prereq_grade_count = create_single_count_table(df_prereq, df_structure)
        prereq_disc_dist = DiscreteDistribution(conv_to_dict(create_single_prob_table(df_prereq_grade_count)))
        state_list.append(State(df_course_data.columns[i], prereq_disc_dist))

    return state_list

# Returns a pomegranate discrete distribution with a generated dictionary based on number of grades
def create_disc_dist(num_grades):
    keys = list(map(str, range(0, num_grades)))
    values = [1/num_grades] * num_grades
    disc_dist_dict = dict(zip(keys, values))

    return DiscreteDistribution(disc_dist_dict)


# Creates a DataFrame of with a column for every possible grade and a column for count
def create_disc_dist_structure(num_grades):
    return pd.DataFrame({'grade': range(0, num_grades)})


# Returns a DataFrame of grades as explained in create_disc_dist_structure along with a column of counts of each event
# as found in the dataframe input data
def create_single_count_table(dataframe, df_structure):
    df_grade_count = df_structure[0:0]

    # Removes nan data
    for i in range(0, len(dataframe.index)):
        if dataframe.iloc[i, 0] != 'nan':
            df_grade_count.loc[len(df_grade_count)] = [dataframe.iloc[:, 0].values[i]]

    df_grade_count = df_grade_count.groupby(df_grade_count.columns.tolist()).size().reset_index(name='count')

    # Makes all values in both DataFrames strings to prevent merge issues
    df_structure = df_structure.astype(str)
    df_grade_count = df_grade_count.astype(str)

    # Gives identical header names to both DataFrames to make merging easier
    headers = list(map(str, range(0, len(df_structure.columns))))
    df_structure.columns = headers
    df_grade_count.columns = headers + ['count']

    # Merges the grade count data into the formatted df_structure
    df_structured_counts = df_structure.merge(df_grade_count, on=headers, how='left')

    # Converts NaN values in the counts to their appropriate value of 0
    df_structured_counts['count'] = df_structured_counts['count'].fillna('0')
    print(df_structured_counts)

    return df_structured_counts


# Turns a grade count table into a discrete distribution table, returns a DataFrame
def create_single_prob_table(df_count_table):
    df_prob_table = df_count_table
    df_prob_table[['count']] = df_prob_table[['count']].astype(float)

    count_sum = df_prob_table['count'].sum()
    df_prob_table.loc[:, 'count'] *= 1 / count_sum

    df_prob_table.rename(columns={'count': 'probability'}, inplace=True)
    return df_prob_table


