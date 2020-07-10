"""
__Author__: Nick Tiede

__Purpose__: An interface to our noisy-avg Bayesian network and the standard Bayesian network. The standard Bayesian
             network will break if it tries to predict when it hasn't seen any instances of one of the input grades
             in the training set.
"""
import copy
from Summer_2020.bn_noisy_avg_predict import create_navg_bn
from Summer_2020.bn_predict import create_std_bn
from Summer_2020.csv_read_write import read_data_csv
from Summer_2020.noisy_avg_calc import create_target_cpt


# Returns a DataFrame of a noisy-avg CPT given a DataFrame of grade data
def create_navg_cpt(df_data, num_grades=11):
    return create_target_cpt(df_data, num_grades)


# Creates and returns a bayesian network model to be predicted from
# Current valid model types are 'noisyavg' and 'standard'
# num_grades options - Standard: 11, whole letter grade: 5, binary: 2
def create_bayesian_network(data_file, num_grades=11, model_type='noisyavg', df_cpt=None):
    if model_type == 'noisyavg':
        return create_navg_bn(data_file, num_grades=num_grades, df_cpt=df_cpt)
    elif model_type == 'standard':
        return create_std_bn(data_file, num_grades=num_grades)
    else:
        print('Bayesian network type not valid: Use model_type= \'standard\' or \'noisyavg\'')
    return


# Wrapper function for pomegranate predict method
# prereq_grade_list should be a list of strings of decimalized grades
# Returns a single predicted decimalized target course grade as a string
def bn_predict(bn_model, prereq_grade_list):
    grade_list = copy.deepcopy(prereq_grade_list.copy())
    grade_list.append(None)
    return str(bn_model.predict([grade_list])[-1][-1])


"""
# Test code
DATA_FILE = 'data\\oops2data.csv'

temp_df_data = read_data_csv(DATA_FILE)
temp_cpt = create_navg_cpt(temp_df_data)
model = create_bayesian_network(DATA_FILE, df_cpt=temp_cpt)

# 0  1  2  3  4  5  6  7  8  9  10
# F  D  D+ C- C  C+ B- B  B+ A- A
print(bn_predict(model, ['7', '7', '7']))
print("\n\n")
"""
