"""
__Author__: Nick Tiede

__Purpose__: To use our Noisy-Avg Bayesian Network methods to create conditional probability table DataFrames based
             on given data to reduce required sample size. The results of this file replace the need to use
             the fitting functions in pomegranate. These tables are meant to be converted to pomegranate
             ConditionalProbabilityTables along with given data DiscreteDistributions to create the Bayesian
             Network model structure for prediction using pomegranate predict methods.

             The Noisy-Avg method is described in this document:
             https://drive.google.com/file/d/1_w_XXjXCFvSzC1LVGFulMcqmTaicVfwB/view?usp=sharing
"""

import pandas as pd
from joblib import Parallel, delayed
from Summer_2020.cartesian_table_creator import create_cartesian_table
from Summer_2020.con_prob_table_creator import create_cpt


# Takes in a pandas DataFrame of course data assuming that the target course is in the last column
# and returns a DataFrame of a conditional probability table using our noisy-avg
def create_target_cpt(dataframe, num_grades):
    num_prereqs = len(dataframe.columns) - 1

    # If a course only has one prereq, the noisy-avg behaves identical to a standard Bayesian network
    # Using the standard BN function speeds up the process
    if num_prereqs == 1:
        return create_cpt(dataframe, num_grades, num_prereqs)

    # Creates the auxiliary node conditional probability table structure without the probability column
    aux_cpt_structure = create_cartesian_table(num_grades, 2)

    # Creates the target node conditional probability table structure
    final_cpt_structure = create_cartesian_table(num_grades, num_prereqs+1)
    final_cpt_structure.columns = [*final_cpt_structure.columns[:-1], 'target']
    final_cpt_structure['probability'] = float(0)

    # Creates a DataFrame with the averages of each combination of auxiliary grades
    df_averages = create_avg_table(create_cartesian_table(num_grades, num_prereqs), num_prereqs)

    # This removes a warning about changing a value of a copy of a DataFrame.
    # This is done in the next code block in an acceptable use case for a temporary DataFrame.
    pd.options.mode.chained_assignment = None

    # Creates a list of each prereq to target course grade count table
    grade_count_table_list = []
    for i in range(0, num_prereqs):
        temp_count_table = create_count_table(dataframe.iloc[:, [i, -1]], aux_cpt_structure, num_grades)
        temp_count_table.columns = ['prereq' + str(i), 'target', 'count']
        grade_count_table_list.append(temp_count_table)

    # Turns the count tables into auxiliary node conditional probability table DataFrames
    aux_cpt_list = []
    for i, item in enumerate(grade_count_table_list):
        temp_aux_cpt = create_aux_cpt(item, num_grades)
        temp_aux_cpt.columns = ['prereq' + str(i), 'target', 'prob' + str(i)]
        aux_cpt_list.append(temp_aux_cpt)

    # Combines the aux CPTs into one DataFrame with the following structure
    # prereq target prob0 prob1 ... probn
    #    0      0     0.1   0.1       0.2
    #    0      1     0.2   0.1       0.3
    #   ...    ...    ...   ...       ...
    #    10     10    0.4   0.3       0.5
    # where the probability columns are the probability of getting the target column grade
    # given the prereq column grade
    combined_aux_cpt = create_combined_cpt(aux_cpt_list)

    target_cpt = create_noisy_avg_cpt(final_cpt_structure, combined_aux_cpt, df_averages, num_prereqs, num_grades)

    return target_cpt


# Returns a DataFrame with counts of prereq to target course grade instances in a CPT like structure
def create_count_table(dataframe, df_structure, num_grades):
    df_grades_filtered = df_structure[0:0]

    # Puts each non empty grade data into a new DataFrame that can be counted
    for i in range(0, len(dataframe.index)):
        # Creates a template row
        filtered_data_row = [0, 0]

        # Filters out nan data
        if dataframe.iloc[i, 0] != 'nan' and dataframe.iloc[i, 1] != 'nan':
            filtered_data_row[0] = int(dataframe.iloc[i, 0])
            filtered_data_row[1] = int(dataframe.iloc[i, 1])
            df_grades_filtered.loc[len(df_grades_filtered)] = filtered_data_row

    # Condenses the rows of the filtered grade dataframe so duplicates are only listed once and a new
    # column is added for the counts of each instance of data
    df_grades_filtered = df_grades_filtered.groupby(df_grades_filtered
                                                    .columns.tolist()).size().reset_index(name='count')

    # Gives identical header names to both DataFrames to make merging easier
    headers = list(map(str, range(0, len(df_structure.columns))))
    df_structure.columns = headers
    df_grades_filtered.columns = headers + ['count']

    # Makes all values in both DataFrames strings to prevent merge issues
    df_structure = df_structure.astype(str)
    df_grades_filtered = df_grades_filtered.astype(str)

    # Merges the grade count data into the full truth table
    df_counts = df_structure.merge(df_grades_filtered, on=headers, how='left')

    # Converts NaN values in the counts to their appropriate value of 0
    df_counts['count'] = df_counts['count'].fillna('0')

    return df_counts


# Turns a grade count table into a conditional probability table for each prereq to target, returns a DataFrame
def create_aux_cpt(df_count_table, num_grades):
    df_prob_table = df_count_table.copy()
    df_prob_table[['count']] = df_prob_table[['count']].astype(float)

    for prereq_grade in range(0, num_grades):
        count_sum = df_prob_table[df_prob_table.iloc[:, 0] == str(prereq_grade)]['count'].sum()
        if count_sum != 0:
            df_prob_table.loc[df_prob_table.iloc[:, 0] == str(prereq_grade), 'count'] *= 1/count_sum

    # Adds some 'fuzz' to the probabilities so none are zero
    # This technically makes each grade probabilities not sum to 1, but this will be normalized later
    df_prob_table.iloc[:, -1] += 0.00001

    df_prob_table.rename(columns={'count': 'probability'}, inplace=True)

    return df_prob_table


# Creates a DataFrame similar to the final CPT structure with a column of averages of grades instead of probabilities
# This gives a slight preference towards higher grades due to rounding.
def create_avg_table(df_cpt_structure, num_prereqs):
    df_averages = df_cpt_structure.astype(float)

    headers = []
    for i in range(0, num_prereqs):
        headers.append('aux' + str(i))

    df_averages.columns = headers

    df_averages['average'] = df_averages.mean(axis=1)
    df_averages['average'] = df_averages['average'].round()

    return df_averages.astype(int).astype(str)


# Takes in a list of auxiliary node CPTs and merges them into a format useful for our noisy-avg algorithm
def create_combined_cpt(aux_cpt_list):
    combined_cpt = aux_cpt_list[0].filter(['prereq0', 'target'], axis=1)
    combined_cpt.columns = ['prereq', 'target']

    for i, item in enumerate(aux_cpt_list):
        combined_cpt['prob' + str(i)] = item.iloc[:, -1]

    return combined_cpt


# Sub function of create_noisy_avg_cpt for parallelization
def create_event_prob(row_index, noisy_avg_cpt, aux_probabilities, df_averages, num_prereqs):
    return calculate_target_prob(list(noisy_avg_cpt.iloc[row_index, 0:num_prereqs]),
                                 str(noisy_avg_cpt.iloc[row_index]['target']),
                                 aux_probabilities, df_averages, num_prereqs)


# Creates the final normalized noisy-avg cpt for the target course
def create_noisy_avg_cpt(cpt_structure, aux_probabilities, df_averages, num_prereqs, num_grades):
    noisy_avg_cpt = cpt_structure.copy()
    noisy_avg_cpt[['probability']] = noisy_avg_cpt[['probability']].astype(float)

    print('Total tasks to complete: ' + str(len(noisy_avg_cpt.index)))

    prob_value_list = Parallel(n_jobs=-1, verbose=True)(delayed(create_event_prob)(i,
                                                                                   noisy_avg_cpt,
                                                                                   aux_probabilities,
                                                                                   df_averages,
                                                                                   num_prereqs)
                                                        for i in range(0, len(noisy_avg_cpt.index)))

    for row_index in range(len(prob_value_list)):
        noisy_avg_cpt.iat[row_index, -1] = prob_value_list[row_index]

    norm_noisy_avg_cpt = normalize_cpt(noisy_avg_cpt,num_prereqs, num_grades)

    return norm_noisy_avg_cpt


# Calculates an individual probability for one cpt event using our noisy-avg method
def calculate_target_prob(prereq_grade_list, target_grade, aux_probabilities, df_averages, num_prereqs):
    target_avg_table = search_avg(df_averages, target_grade)
    target_avg_table['probability'] = float(0)
    target_avg_table.reset_index(drop=True, inplace=True)

    for row_index in range(0, len(target_avg_table.index)):
        target_avg_table.iat[row_index, -1] = calculate_aux_combination(prereq_grade_list, aux_probabilities,
                                                                        list(target_avg_table.iloc[row_index,
                                                                             0:num_prereqs]))

    return target_avg_table['probability'].sum()


# Searches through a DataFrame of auxiliary value averages and returns a DataFrame of the rows that match the input
def search_avg(df_averages, target_grade):
    return df_averages.loc[df_averages['average'] == str(target_grade)].copy()


# Returns the multiplication of the probabilities associated with an input grade list
def calculate_aux_combination(prereq_grade_list, aux_probs, aux_grade_list):
    prob_mult = 1
    for i in range(0, len(prereq_grade_list)):
        prob_mult *= float(aux_probs.loc[(aux_probs.iloc[:, 0] == prereq_grade_list[i]) &
                                     (aux_probs.iloc[:, 1] == aux_grade_list[i])]['prob' + str(i)])

    return prob_mult


# Normalizes the probabilities in a CPT based on the number of grades
# The input CPT must be ordered like the cartesian product of the grades
# This requirement speeds up the process significantly
def normalize_cpt(noisy_avg_cpt, num_prereqs, num_grades):
    # This is not a deep copy for memory reasons. Add .copy() at the end to be able to compare to original.
    norm_noisy_avg_cpt = noisy_avg_cpt

    for i in range(num_grades ** num_prereqs):
        row_i_min = int(i*num_grades)
        row_i_max = int((i*num_grades) + num_grades)
        prob_sum = norm_noisy_avg_cpt.iloc[row_i_min:row_i_max, -1].sum()
        norm_modifier = 1/prob_sum
        norm_noisy_avg_cpt.iloc[row_i_min:row_i_max, -1] *= norm_modifier

    return norm_noisy_avg_cpt
