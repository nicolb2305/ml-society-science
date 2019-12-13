import numpy as np
from sklearn.linear_model import LogisticRegression
from random_recommender import RandomRecommender

class HistoricalRecommender(RandomRecommender):
    def fit_treatment_outcome(self, data, actions, outcome):
        print("Fitting treatment outcomes")
        self.model = LogisticRegression(solver='sag')
        self.model.fit(data, actions[:, 0])
        return None

    def recommend(self, user_data):
        user_data = np.array([user_data])
        return self.model.predict(user_data)[0]
