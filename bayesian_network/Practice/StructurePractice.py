from pomegranate import BayesianNetwork
import seaborn, time
seaborn.set_style('whitegrid')
import numpy

X = numpy.random.randint(2, size=(2000, 7))
X[:,3] = X[:,1]
X[:,6] = X[:,1]

X[:,0] = X[:,2]

X[:,4] = X[:,5]

model = BayesianNetwork.from_samples(X, algorithm='exact')
print(model.structure)

print(model.predict([[None, None, None, None, None, None, None]]))