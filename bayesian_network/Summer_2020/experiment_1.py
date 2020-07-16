"""
__Author__: Nick Tiede

__Purpose__: Generate noisy-or CPTs and save them for later use in experiments. This is hard coded for the first
             experiment.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from bayesian_network.Summer_2020.bn_interface import *

AIS_DATA_FILE = 'data\\AIS.csv'
COBOL_DATA_FILE = 'data\\COBOL.csv'
SAD_DATA_FILE = 'data\\SAD.csv'
SDI_DATA_FILE = 'data\\SDI.csv'
SAVE_LOC = 'cpt\\Exp1 '

generate_navg_cpt(AIS_DATA_FILE, SAVE_LOC)
generate_navg_cpt(COBOL_DATA_FILE, SAVE_LOC)
generate_navg_cpt(SAD_DATA_FILE, SAVE_LOC)
generate_navg_cpt(SDI_DATA_FILE, SAVE_LOC)
