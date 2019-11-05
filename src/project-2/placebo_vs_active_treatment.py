import numpy as np
import pandas
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes


X = pandas.read_csv('data/medical/historical_X.dat', header=None, sep=" ")
A = pandas.read_csv('data/medical/historical_A.dat', header=None, sep=" ").values
Y = pandas.read_csv('data/medical/historical_Y.dat', header=None, sep=" ").values

n_clusters = 2
km = KModes(n_clusters=n_clusters)
clusters = km.fit(X)
predictions = clusters.predict(X)

all_data = pandas.crosstab(A[:, 0],
                           Y[:, 0],
                           rownames=['action'],
                           colnames=['outcome'],
                           normalize='index')

print('Action-outcome contingency for all the data:', all_data, sep='\n', end='\n\n')
clustered_data = []
for i in range(n_clusters):
    clustered_data.append(pandas.crosstab(A[predictions == i, 0],
                                          Y[predictions == i, 0],
                                          rownames=['action'],
                                          colnames=['outcome'],
                                          normalize='index'))
    print(f'Action-outcome contingency for cluster {i}:', clustered_data[-1], sep='\n', end='\n\n')
    cluster_frequencies = clustered_data[i].to_numpy().reshape((1, 4))[0]
    all_data_frequencies = (all_data*np.sum(predictions == i)/10000).round(0).to_numpy().reshape((1, 4)).astype(int)[0]
