"""
__Author__: Nick Tiede

__Purpose__: To be used as a speed test to compare computers for generating noisy-avg CPTs
"""

from bayesian_network.Summer_2020.bn_interface import generate_navg_cpt

OOPS2_DATA_FILE = 'data\\oops2data.csv'
SAVE_LOC = 'cpt\\Speed Test '

generate_navg_cpt(OOPS2_DATA_FILE, SAVE_LOC)
