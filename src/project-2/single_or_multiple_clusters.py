import numpy as np
import pandas
from scipy.stats import chisquare
from kmodes.kmodes import KModes

X = pandas.read_csv('data/medical/historical_X.dat', header=None, sep=" ")

n_clusters = 2
km = KModes(n_clusters=n_clusters)
clusters = km.fit(X[X.columns[0:128]])
predictions = clusters.predict(X[X.columns[0:128]])

all_data = pandas.crosstab(X[X.columns[128]], X[X.columns[129]])

print('Symptoms contingency in full data set:', all_data, sep='\n', end='\n\n')
clustered_data = []
for i in range(n_clusters):
    clustered_data.append(pandas.crosstab(X[predictions == i][128],
                                          X[predictions == i][129]))
    print(f'Symptoms present in cluster {i}:', clustered_data[-1], sep='\n', end='\n\n')
    cluster_frequencies = clustered_data[i].to_numpy().reshape((1, 4))[0]
    all_data_frequencies = (all_data*np.sum(predictions == i)/10000).round(0).to_numpy().reshape((1, 4)).astype(int)[0]
    print(f'Chi square test between symptoms contingency in all data and cluster {i}: p={chisquare(cluster_frequencies, all_data_frequencies)[1]:.3e}')
