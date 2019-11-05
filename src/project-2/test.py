import numpy as np
import pandas
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, chisquare
from kmodes.kmodes import KModes

X = pandas.DataFrame(np.random.choice(2, (10000, 130)))

n_clusters = 2
km = KModes(n_clusters=n_clusters)
print(X[X.columns[0:128]])
clusters = km.fit(X[X.columns[0:128]])
#print(clusters.cluster_centroids_)
predictions = clusters.predict(X[X.columns[0:128]])
print(np.sum(predictions == 0))

all_data = pandas.crosstab(X[X.columns[128]], X[X.columns[129]])

print(all_data)
clustered_data = []
for i in range(n_clusters):
    clustered_data.append(pandas.crosstab(X[predictions == i][128],
                                          X[predictions == i][129]))
    print(clustered_data[-1])
    cluster_frequencies = clustered_data[i].to_numpy().reshape((1, 4))[0]
    all_data_frequencies = (all_data*np.sum(predictions == i)/10000).round(0).to_numpy().reshape((1, 4)).astype(int)[0]
    print(chisquare(cluster_frequencies, all_data_frequencies))
