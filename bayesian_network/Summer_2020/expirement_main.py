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

# Read in the data
ais_df = read_data_csv(AIS_DATA_FILE)
cobol_df = read_data_csv(COBOL_DATA_FILE)
sad_df = read_data_csv(SAD_DATA_FILE)
sdi_df = read_data_csv(SDI_DATA_FILE)

# Create the CPTs as DataFrames
ais_cpt = create_navg_cpt(ais_df)
cobol_cpt = create_navg_cpt(cobol_df)
sad_cpt = create_navg_cpt(sad_df)
sdi_cpt = create_navg_cpt(sdi_df)

# Save the CPTs as CSV files
save_cpt_as_csv(ais_cpt, 'cpt\\AIScpt.csv')
print('AIS CPT saved')
save_cpt_as_csv(cobol_cpt, 'cpt\\COBOLcpt.csv')
print('COBOL CPT saved')
save_cpt_as_csv(sad_cpt, 'cpt\\SADcpt.csv')
print('SAD CPT saved')
save_cpt_as_csv(sdi_cpt, 'cpt\\SDIcpt.csv')
print('SDI CPT saved')

"""
# Test code
DATA_FILE = 'data\\oops2data.csv'

temp_df_data = read_data_csv(DATA_FILE)
temp_cpt = create_navg_cpt(temp_df_data)
model = create_bayesian_network(temp_df_data, df_cpt=temp_cpt)

# 0  1  2  3  4  5  6  7  8  9  10
# F  D  D+ C- C  C+ B- B  B+ A- A
print(bn_predict(model, ['7', '7', '7']))
print("\n\n")
"""