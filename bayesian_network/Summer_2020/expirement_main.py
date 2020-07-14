"""
__Author__: Nick Tiede

__Purpose__: Generate noisy-or CPTs and save them for later use in experiments. This is hard coded for the first
             experiment.
"""

from Summer_2020.bn_interface import *

AIS_DATA_FILE = 'data\\AIS.csv'
COBOL_DATA_FILE = 'data\\COBOL.csv'
SAD_DATA_FILE = 'data\\SAD.csv'
SDI_DATA_FILE = 'data\\SDI.csv'
SAVE_LOC = 'cpt\\'

generate_navg_cpt(AIS_DATA_FILE, SAVE_LOC)
generate_navg_cpt(COBOL_DATA_FILE, SAVE_LOC)
generate_navg_cpt(SAD_DATA_FILE, SAVE_LOC)
generate_navg_cpt(SDI_DATA_FILE, SAVE_LOC)

"""
# Test code

# 0  1  2  3  4  5  6  7  8  9  10
# F  D  D+ C- C  C+ B- B  B+ A- A

generate_navg_cpt(SAD_DATA_FILE, SAVE_LOC)
df_data = load_data_csv(SAD_DATA_FILE)
df_cpt = load_cpt_from_csv('cpt\\Systems Analysis and Design CPT')
model = create_bayesian_network(df_data, df_cpt=df_cpt)
print('Given 7,7 Predicted: ' + bn_predict(model, ['7', '7']))
print('Given 10,10 Predicted: ' + bn_predict(model, ['10', '10']))
print('Given 8,5 Predicted: ' + bn_predict(model, ['8', '5']))
print('Given 2,4 Predicted: ' + bn_predict(model, ['2', '4']))
print('Given 0,0 Predicted: ' + bn_predict(model, ['0', '0']))
print('Done')
"""