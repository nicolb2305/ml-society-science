from improved_recommender import ImprovedRecommender
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample

X = pandas.read_csv('data/medical/historical_X.dat', header=None, sep=" ").values
A = pandas.read_csv('data/medical/historical_A.dat', header=None, sep=" ").values
Y = pandas.read_csv('data/medical/historical_Y.dat', header=None, sep=" ").values
AY = np.column_stack((A, Y))
recommender = ImprovedRecommender(2, 2)

n_samples = 10000
utilities = np.zeros((n_samples))
for i in range(n_samples):
    sample = resample(AY, replace=True)
    utilities[i] = recommender.estimate_utility(None, sample[:, 0], sample[:, 1])
quantiles = np.quantile(utilities, [0.025, 0.975])
print(f"Estimated utility: {recommender.estimate_utility(X, A, Y)}")
print(f"95% error bounds: {quantiles}")
plt.hist(utilities, bins=30)
plt.title("Utilities from 10000 bootstrap samples")
plt.xlabel("Total utility per sample")
plt.ylabel("Occurrences")
plt.axvline(quantiles[0], color='black', ls='--')
plt.axvline(quantiles[1], color='black', ls='--')
plt.savefig("utility_bootstrap.png")
