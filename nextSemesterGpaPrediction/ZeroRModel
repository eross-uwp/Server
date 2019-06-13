import numpy as np
import pandas as pd
import io
import random

if __name__== "__main__":
    __minTermNumber = 2
    __combinedDataRange = None

    idNNumOfTerm = pd.read_csv("Book1.csv")

    __combinedDataRange = idNNumOfTerm['numOfTerm'][1]

    for i in range(__combinedDataRange, 1, -1):
        ids = np.zeros(0)
        for j in range(idNNumOfTerm['numOfTerm'].size):
            if  idNNumOfTerm['numOfTerm'][j] > 1:
                ids = np.append(ids, random.randint(__minTermNumber, idNNumOfTerm['numOfTerm'][j]))
                #ids = np.append(ids, random.randint(__minTermNumber, 1))
        #del ids
        print("")

    print(ids)
    print()