import pandas as pd
import numpy as np
import Lists
import csv

if __name__ == "__main__":
    data = pd.read_csv("GradeTest.csv")
    data.head()

    print(data.iloc[0][0])

    for y in range(13):
        row = data.get
        rowList = []
        rowList.append(row[0])
        for x in range(1, len(row) - 1):
            if not np.isnan(row[x]) and not np.isnan(row[x + 1]):
                rowList.append((row[x], row[x + 1]))

    print(rowList)

    pairData = pd.DataFrame([])
    pairData.append(np.asarray(rowList))
    print(pairData)







# for each student choose random pair
# have at least two semesters