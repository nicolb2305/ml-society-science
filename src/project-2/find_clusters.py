import numpy as np
from kmodes.kmodes import KModes
import pandas
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import silhouette_samples, silhouette_score

X = pandas.read_csv('data/medical/historical_X.dat', header=None, sep=" ").values

X_train, X_test = train_test_split(X, test_size=0.5)
for k in range(1, 6):
    km = KModes(n_clusters=k)
    clusters = km.fit(X_train)
    predictions = clusters.predict(X_test)
    ssw = 0
    for i in range(len(X_test)):
        temp = X_test[i] - clusters.cluster_centroids_[predictions[i]]
        ssw += np.dot(temp, temp)
    d = X_test.shape[1]
    n = X_test.shape[0]
    print(d*np.log(np.sqrt(ssw/(d*n**2)))+np.log(k))


"""
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

linked = linkage(features[:100], metric='cosine')
dendrogram(linked,
            orientation='top',
            labels=range(len(features[:100])),
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()
"""
