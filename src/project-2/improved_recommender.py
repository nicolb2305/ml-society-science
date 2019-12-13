from policy import Policy
import numpy as np
import pandas
from sklearn.linear_model import LogisticRegression
from random_recommender import RandomRecommender

class ImprovedRecommender(RandomRecommender):
    def _default_reward(self, action, outcome):
        return -0.1*(action != 0) + outcome

    def fit_treatment_outcome(self, data, actions, outcome):
        print("Fitting treatment outcomes")
        self.models = []
        for action in range(self.n_actions):
            self.models.append(LogisticRegression(solver='sag'))
            action_data = (actions == action).ravel()
            self.models[-1].fit(data[action_data], outcome[action_data].ravel())
        return None

    def estimate_utility(self, data, actions, outcome, policy=None):
        if not policy:
            return np.sum(self.reward(actions, outcome))/len(actions)
        else:
            pi_hat = actions.value_counts()/actions.size
            recommendations = np.zeros(actions.size)
            for i in range(actions.size):
                recommendations[i] = policy.get_best_action(data.iloc[i], [self.predict_proba(data.iloc[i], a) for a in range(self.n_actions)])
            return np.sum(self.reward(recommendations, outcome) * recommendations / actions.map(pi_hat))/len(actions)

    def predict_proba(self, data, treatment):
        return self.models[treatment].predict_proba(pandas.DataFrame(data).transpose())

    def get_action_probabilities(self, user_data):
        measurable_effect_proba = []
        for action in range(self.n_actions):
            measurable_effect_proba.append(self.predict_proba(user_data, action)[0][1])
        probas = measurable_effect_proba
        return_probas = np.zeros(self.n_actions)
        return_probas[np.argmax(probas)] = 1
        return return_probas
