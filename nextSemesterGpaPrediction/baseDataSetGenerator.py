import pandas as pd
import zeroRModel as zrm
import random

if __name__ == "__main__":
    termGPA = pd.read_csv('data\\termGPA.csv', index_col="index")  # our raw dataset
    headers = list(termGPA)

    datasetForPairs = pd.DataFrame()

    for id in headers:  # for every student id in the raw dataset
        terms = []
        for termNumber in termGPA[id].index:  # for every term row of an id column
            if termGPA[id].loc[termNumber] > 0:
                terms.append(termNumber)  # copy all term numbers that relate to a non-zero gpa to a temp array
        if len(terms) > 1:  # as long as a student id is related to at least 2 terms of non-zero gpa
            randomTermIndex = random.randrange(1, len(terms))
            secondTerm = terms[randomTermIndex]
            firstTerm = terms[randomTermIndex - 1]
            datasetForPairs[id] = [firstTerm, secondTerm]  # save a random term and the term previous to it for each id

    finalDataSetHeaders = ['id', 'prev term number', 'current term number', 'prev GPA', 'current GPA']
    finalDataSet = pd.DataFrame(columns=finalDataSetHeaders)

    ids = list(datasetForPairs)

    # for each student id, copy the 2 terms we found and the corresponding GPAs to the final dataset
    for id in ids:
        finalDataSet = finalDataSet.append({'id': id, 'prev term number': datasetForPairs[id][0],
                                            'current term number': datasetForPairs[id][1],
                                            'prev GPA': termGPA[id][datasetForPairs[id][0]],
                                            'current GPA': termGPA[id][datasetForPairs[id][1]]}, ignore_index=True)

    # print(finalDataSet)
    finalDataSet.to_csv('data\\finalDataSet.csv', encoding='utf-8', index=False)

    print(zrm.predict(finalDataSet['prev GPA']))
