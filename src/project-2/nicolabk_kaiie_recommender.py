from sklearn import linear_model
from sklearn.utils import resample
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

class NicolabkKaiieRecommender:
    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

    def _default_reward(self, action, outcome):
        return -0.1*action + outcome

    def set_reward(self, reward):
        self.reward = reward

    def fit_data(self, data):
        print("Preprocessing data")
        return None

    def fit_treatment_outcome(self, data, actions, outcome):
        print("Fitting treatment outcomes")

        class Policy:
            def __init__(self, model_placebo, model_treatment):
                self.model_placebo = model_placebo
                self.model_treatment = model_treatment

            def fit_models(self, data, actions, outcome):
                placebo_data = (actions == 0).to_numpy()
                treatment_data = (actions == 1).to_numpy()
                self.model_placebo.fit(data[placebo_data], outcome[placebo_data].ravel())
                self.model_treatment.fit(data[treatment_data], outcome[treatment_data].ravel())

            # Predict probability of person seeing a notable effect given placebo, and given treatment
            def predict_proba(self, data, treatment):
                # Check if data contains one person or multiple
                temp = False
                if type(data) == pandas.Series:
                    data = data.to_frame().transpose()
                    temp = True

                if treatment == 0:
                    return_value = self.model_placebo.predict_proba(data)
                elif treatment == 1:
                    return_value = self.model_treatment.predict_proba(data)

                if temp:
                    return return_value[0]
                else:
                    return return_value

            def get_action_probability(self, data):
                return None

        model = Policy(LogisticRegression(solver='sag'), LogisticRegression(solver='sag'))
        model.fit_models(data, actions, outcome)
        self.model = model
        return model

    def estimate_utility(self, data, actions, outcome, policy=None):
        if not policy:
            return np.sum(self.reward(actions, outcome))
        else:
            pi_hat = actions.value_counts()/actions.size
            recommendations = np.zeros(actions.size)
            for i in range(actions.size):
                placebo_effect_proba = policy.predict_proba(data.iloc[i], 0)
                treatment_effect_proba = policy.predict_proba(data.iloc[i], 1)
                placebo_utility = 0*placebo_effect_proba[0] + 1*placebo_effect_proba[1]
                treatment_utility = -0.1*treatment_effect_proba[0] + 0.9*treatment_effect_proba[1]
                recommendations[i] = np.argmax(np.column_stack((placebo_utility, treatment_utility)), axis=1)
            return np.sum(outcome * recommendations / actions.map(pi_hat))#.mean()

    def predict_proba(self, data, treatment):
        return numpy.zeros(self.n_outcomes)

    def get_action_probabilities(self, user_data):
        print("Recommending")
        return None

    def recommend(self, user_data):
        return np.random.choice(self.n_actions, p = self.get_action_probabilities(user_data))

    def observe(self, user, action, outcome):
        return None

    def final_analysis(self):
        return None


if __name__ == "__main__":
    X = pandas.read_csv('data/medical/historical_X.dat', header=None, sep=" ")
    A = pandas.read_csv('data/medical/historical_A.dat', header=None, sep=" ").squeeze()
    Y = pandas.read_csv('data/medical/historical_Y.dat', header=None, sep=" ").squeeze()
    AY = np.column_stack((A, Y))
    recommender = NicolabkKaiieRecommender(2, 2)

    n_samples = 10000
    utilities = np.zeros((n_samples))
    for i in range(n_samples):
        sample = resample(AY, replace=True)
        utilities[i] = recommender.estimate_utility(None, sample[:, 0], sample[:, 1])
    quantiles = np.quantile(utilities, [0.025, 0.975])
    print(f"95% error bounds: {quantiles}")
    plt.hist(utilities, bins=30)
    plt.title("Utilities from 10000 bootstrap samples")
    plt.xlabel("Utility")
    plt.ylabel("Occurrences")
    plt.axvline(quantiles[0], color='black', ls='--')
    plt.axvline(quantiles[1], color='black', ls='--')
    plt.show()

    policy = recommender.fit_treatment_outcome(X, A, Y)
    print(f"Utility of historical data: {recommender.estimate_utility(X, A, Y)}")
    print(f"Utility of historical data with new policy: {recommender.estimate_utility(X, A, Y, policy=policy)}")
