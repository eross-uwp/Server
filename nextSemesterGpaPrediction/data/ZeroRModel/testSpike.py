#Import CSV file from URL that has latest CSO status with location names and status
#This script only displays on the screen. Need to create list data into a file format.
#STEP 1

import csv
import urllib
import urllib.request

#URL address that contains cso status with location name. This includes king county and SPU's data
url = 'https://raw.githubusercontent.com/earos-uwp/Server/master/nextSemesterGpaPrediction/data/test_train/test_1.csv'
webpage = urllib.request.urlopen(url)
datareader = csv.reader(webpage)

import pandas as pd

print(pd.read_csv(webpage))