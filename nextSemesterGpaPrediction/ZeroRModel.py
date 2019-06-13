import numpy as np
import pandas as pd
import io
import random

if __name__== "__main__":
    __minTermNumber = 2
    termGPA = pd.read_csv('termGPA.csv', index_col="index")
    headers = list(termGPA)

    datasetForPairs = pd.DataFrame()

    for id in headers:
        terms = []
        for termNumber in termGPA[id].index:
            if termGPA[id].loc[termNumber] > 0:
                terms.append(termNumber)
        if len(terms) > 1:
            randomTermIndex = random.randrange(1, len(terms))
            secondTerm = terms[randomTermIndex]
            firstTerm = terms[randomTermIndex - 1]
            datasetForPairs[id] = [firstTerm, secondTerm]

    finalDataSetHeaders = ['id', 'prev term number', 'current term number', 'prev GPA', 'current GPA']
    finalDataSet = pd.DataFrame(columns=finalDataSetHeaders)

    ids = list(datasetForPairs)

    for id in ids:
        finalDataSet = finalDataSet.append({'id' : id, 'prev term number' : datasetForPairs[id][0],
                                            'current term number' : datasetForPairs[id][1],
                                            'prev GPA' : termGPA[id][datasetForPairs[id][0]],
                                            'current GPA' : termGPA[id][datasetForPairs[id][1]]}, ignore_index=True)
        # tempAdding = pd.DataFrame('id':[id])
        # finalDataSet['id'] = finalDataSet['id'].append(tempAdding, ignore_index=False)
        # del tempAdding
        #
        # tempAdding = pd.Series([datasetForPairs[id][0]], index = [id])
        # finalDataSet['prev term number'] = finalDataSet['prev term number'].append(tempAdding, ignore_index=False)
        # del tempAdding
        #
        # tempAdding = pd.Series([datasetForPairs[id][1]], index = [id])
        # finalDataSet['current term number'] = finalDataSet['current term number'].append(tempAdding, ignore_index=False)
        # del tempAdding
        #
        # tempAdding = pd.Series([termGPA[id][datasetForPairs[id][0]]], index = [id])
        # finalDataSet['prev GPA'] = finalDataSet['prev GPA'].append(tempAdding, ignore_index=False)
        # del tempAdding
        #
        # tempAdding = pd.Series([termGPA[id][datasetForPairs[id][1]]], index = [id])
        # finalDataSet['prev GPA'] = finalDataSet['prev GPA'].append(tempAdding, ignore_index=False)
        #
        # tempAdding = pd.DataFrame('id':[id])
        # del tempAdding

    print(finalDataSet)
