from improved_recommender import ImprovedRecommender
import pandas
from policy import Policy

X = pandas.read_csv('data/medical/historical_X.dat', header=None, sep=" ")
A = pandas.read_csv('data/medical/historical_A.dat', header=None, sep=" ").squeeze()
Y = pandas.read_csv('data/medical/historical_Y.dat', header=None, sep=" ").squeeze()

recommender = ImprovedRecommender(2, 2)
recommender.fit_treatment_outcome(X, A, Y)
print(f"Utility of historical data: {recommender.estimate_utility(X, A, Y)}")
print(f"Estimated utility of historical data with new policy: {recommender.estimate_utility(X, A, Y, policy=Policy())}")
