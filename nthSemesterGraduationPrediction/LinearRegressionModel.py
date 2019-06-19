# Using linear regression to predict if a student will graduate or not

'''
___authors___: Austin FitzGerald
'''

from sklearn import metrics
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np



if __name__ == "__main__":
    stratify_fold()