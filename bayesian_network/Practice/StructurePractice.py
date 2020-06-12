from pomegranate import BayesianNetwork
import seaborn, time
seaborn.set_style('whitegrid')
import numpy

# This is from a practice example located here
# https://github.com/jmschrei/pomegranate/blob/master/tutorials/B_Model_Tutorial_4b_Bayesian_Network_Structure_Learning.ipynb

X = numpy.random.randint(2, size=(2000, 7))
print(type(X))

X[:,3] = X[:,1]
X[:,6] = X[:,1]

X[:,0] = X[:,2]

X[:,4] = X[:,5]

model = BayesianNetwork.from_samples(X, algorithm='exact')

print("\n")
print(model.structure)

print(model.predict([[None, None, None, None, None, None, None]]))

print("\n")
print(X)