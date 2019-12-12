import numpy as np
import pandas
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes

X = pandas.read_csv('data/medical/historical_X.dat', header=None, sep=" ")

distances = []

for n_clusters in range(1, 6):
    km = KModes(n_clusters=n_clusters)

    clusters = km.fit(X)
    predictions = clusters.predict(X)

    distance = np.mean(np.invert(X.to_numpy() == clusters.cluster_centroids_[predictions]), axis=1)
    distances.append(np.mean([np.mean(distance[predictions == i]) for i in range(n_clusters)]))

plt.scatter(range(1, 6), distances)
plt.show()
