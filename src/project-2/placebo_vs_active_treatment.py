import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from kmodes.kmodes import KModes


X = pandas.read_csv('data/medical/historical_X.dat', header=None, sep=" ")
A = pandas.read_csv('data/medical/historical_A.dat', header=None, sep=" ").values
Y = pandas.read_csv('data/medical/historical_Y.dat', header=None, sep=" ").values

n_clusters = 2
km = KModes(n_clusters=n_clusters)
clusters = km.fit(X)
predictions = clusters.predict(X)

"""
print(X[(X[128] == 0) & (X[129] == 0)])
print(pandas.crosstab(A[(X[128] == 0) & (X[129] == 0)].to_numpy()[:, 0],
                      Y[(X[128] == 0) & (X[129] == 0)].to_numpy()[:, 0],
                      rownames=['action'],
                      colnames=['outcome'],
                      normalize='index'))
print(pandas.crosstab(A[(X[128] == 0) & (X[129] == 1)].to_numpy()[:, 0],
                      Y[(X[128] == 0) & (X[129] == 1)].to_numpy()[:, 0],
                      rownames=['action'],
                      colnames=['outcome'],
                      normalize='index'))
print(pandas.crosstab(A[(X[128] == 1) & (X[129] == 0)].to_numpy()[:, 0],
                      Y[(X[128] == 1) & (X[129] == 0)].to_numpy()[:, 0],
                      rownames=['action'],
                      colnames=['outcome'],
                      normalize='index'))
print(pandas.crosstab(A[(X[128] == 1) & (X[129] == 1)].to_numpy()[:, 0],
                      Y[(X[128] == 1) & (X[129] == 1)].to_numpy()[:, 0],
                      rownames=['action'],
                      colnames=['outcome'],
                      normalize='index'))
"""

all_data = pandas.crosstab(A[:, 0],
                           Y[:, 0],
                           rownames=['action'],
                           colnames=['outcome'],
                           normalize='index')

print(all_data)
clustered_data = []
for i in range(n_clusters):
    clustered_data.append(pandas.crosstab(A[predictions == i, 0],
                                          Y[predictions == i, 0],
                                          rownames=['action'],
                                          colnames=['outcome'],
                                          normalize='index'))
    print(clustered_data[-1])
    cluster_frequencies = clustered_data[i].to_numpy().reshape((1, 4))[0]
    all_data_frequencies = (all_data*np.sum(predictions == i)/10000).round(0).to_numpy().reshape((1, 4)).astype(int)[0]
