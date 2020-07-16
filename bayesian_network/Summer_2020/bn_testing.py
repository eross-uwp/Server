"""
__Author__: Nick Tiede

__Purpose__: This is a place to test bn_interface functions
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from bayesian_network.Summer_2020.bn_interface import *

# 0  1  2  3  4  5  6  7  8  9  10
# F  D  D+ C- C  C+ B- B  B+ A- A

AIS_DATA_FILE = 'data\\AIS.csv'
COBOL_DATA_FILE = 'data\\COBOL.csv'
SAD_DATA_FILE = 'data\\SAD.csv'
SDI_DATA_FILE = 'data\\SDI.csv'
SAVE_LOC = 'cpt\\'

generate_navg_cpt(AIS_DATA_FILE, SAVE_LOC)
generate_navg_cpt(COBOL_DATA_FILE, SAVE_LOC)
generate_navg_cpt(SAD_DATA_FILE, SAVE_LOC)
generate_navg_cpt(SDI_DATA_FILE, SAVE_LOC)


# Tests predictions for 2 prereq course
def test_pred_1(test_model):
    print('Given 10  Predicted: ' + bn_predict(test_model, ['10']))
    print('Given 7   Predicted: ' + bn_predict(test_model, ['7']))
    print('Given 0   Predicted: ' + bn_predict(test_model, ['0']))
    print('Given 9   Predicted: ' + bn_predict(test_model, ['9']))
    print('Given 2   Predicted: ' + bn_predict(test_model, ['2']))
    return


# Tests predictions for 2 prereq course
def test_pred_2(test_model):
    print('Given 10,10 Predicted: ' + bn_predict(test_model, ['10', '10']))
    print('Given 7,7   Predicted: ' + bn_predict(test_model, ['7', '7']))
    print('Given 0,0   Predicted: ' + bn_predict(test_model, ['0', '0']))
    print('Given 9,5   Predicted: ' + bn_predict(test_model, ['9', '5']))
    print('Given 2,4   Predicted: ' + bn_predict(test_model, ['2', '4']))
    return


# Tests predictions for 3 prereq course
def test_pred_3(test_model):
    print('Given 10,10,10 Predicted: ' + bn_predict(test_model, ['10', '10', '10']))
    print('Given 7,7,7    Predicted: ' + bn_predict(test_model, ['7', '7', '7']))
    print('Given 0,0,0    Predicted: ' + bn_predict(test_model, ['0', '0', '0']))
    print('Given 7,5,3    Predicted: ' + bn_predict(test_model, ['7', '5', '3']))
    print('Given 2,4,6    Predicted: ' + bn_predict(test_model, ['2', '4', '6']))
    return


# Applications in Information Systems Test
print('Applications in Information Systems:')
df_data = load_data_csv(AIS_DATA_FILE)
model = create_bayesian_network(df_data, model_type='standard')
test_pred_3(model)
print('Standard Bayesian Network Test Complete')
df_cpt = load_cpt_from_csv('cpt\\Applications in Information Systems CPT')
model = create_bayesian_network(df_data, df_cpt=df_cpt)
test_pred_3(model)
print('Noisy-Avg Bayesian Network Test Complete \n')

# Programming in COBOL Test
print('Programming in COBOL:')
df_data = load_data_csv(COBOL_DATA_FILE)
model = create_bayesian_network(df_data, model_type='standard')
test_pred_1(model)
print('Standard Bayesian Network Test Complete')
df_cpt = load_cpt_from_csv('cpt\\Programming in COBOL CPT')
model = create_bayesian_network(df_data, df_cpt=df_cpt)
test_pred_1(model)
print('Noisy-Avg Bayesian Network Test Complete \n')

# Systems Analysis and Design Test
print('Systems Analysis and Design:')
df_data = load_data_csv(SAD_DATA_FILE)
model = create_bayesian_network(df_data, model_type='standard')
test_pred_2(model)
print('Standard Bayesian Network Test Complete')
df_cpt = load_cpt_from_csv('cpt\\Systems Analysis and Design CPT')
model = create_bayesian_network(df_data, df_cpt=df_cpt)
test_pred_2(model)
print('Noisy-Avg Bayesian Network Test Complete \n')

# Systems Development and Implementation Test
print('Systems Development and Implementation:')
df_data = load_data_csv(SDI_DATA_FILE)
model = create_bayesian_network(df_data, model_type='standard')
test_pred_2(model)
print('Standard Bayesian Network Test Complete')
df_cpt = load_cpt_from_csv('cpt\\Systems Development and Implementation CPT')
model = create_bayesian_network(df_data, df_cpt=df_cpt)
test_pred_2(model)
print('Noisy-Avg Bayesian Network Test Complete \n')
