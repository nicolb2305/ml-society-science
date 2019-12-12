from sklearn import linear_model
import numpy as np
from kmodes.kmodes import KModes
from sklearn.naive_bayes import BernoulliNB
from sklearn.exceptions import NotFittedError

class AdaptiveRecommender:
    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

    def _default_reward(self, action, outcome):
        return outcome

    def set_reward(self, reward):
        self.reward = reward

    def fit_data(self, data):
        print("Preprocessing data")
        self.clusters = KModes(n_clusters=2).fit(data)
        return None

    def fit_treatment_outcome(self, data, actions, outcome):
        self.fit_data(data)
        print("Fitting treatment outcomes")
        self.models = [[], []]
        cluster_predictions = self.clusters.predict(data)
        for cluster in range(2):
            cluster_data = cluster_predictions == cluster
            for action in range(self.n_actions):
                self.models[cluster].append(BernoulliNB())
                action_data = (actions == action).ravel()
                data_indecies = np.logical_and(action_data, cluster_data)
                # Fit model if we have at least 1 data point
                if np.any(data_indecies):
                    self.models[cluster][-1].fit(data[data_indecies], outcome[data_indecies].ravel())

        self.t = data.shape[0]
        self.good = np.zeros((2, self.n_actions))
        self.total = np.zeros((2, self.n_actions))
        predictions = []
        for i in range(len(data)):
            predictions.append(np.argmax(elf.predict_proba(data[i], actions[i])))
        self.total[0][0] += np.sum((predictions == 0 and outcome.flatten() == 0)[cluster_predictions == 0])
        self.total[1][0] += np.sum((predictions == 0 and outcome.flatten() == 0)[cluster_predictions == 0])
        self.total[0][1] += np.sum((predictions == 1 and outcome.flatten() == 1)[cluster_predictions == 1])
        self.total[1][1] += np.sum((predictions == 1 and outcome.flatten() == 1)[cluster_predictions == 1])
        return None

    ## Estimate the utility of a specific policy from historical data (data, actions, outcome),
    ## where utility is the expected reward of the policy.
    ##
    ## If policy is not given, simply use the average reward of the observed actions and outcomes.
    ##
    ## If a policy is given, then you can either use importance
    ## sampling, or use the model you have fitted from historical data
    ## to get an estimate of the utility.
    ##
    ## The policy should be a recommender that implements get_action_probability()
    def estimate_utility(self, data, actions, outcome, policy=None):
        return 0

    # Return a distribution of effects for a given person's data and a specific treatment.
    # This should be an numpy.array of length self.n_outcomes
    def predict_proba(self, data, treatment):
        data = np.array([data])
        cluster = self.clusters.predict(data)[0]
        try:
            proba = self.models[cluster][treatment].predict_proba(data)[0]
        except NotFittedError:
            if treatment > 2 and data[0][treatment-1] != 1:
                proba = [1, 0]
            else:
                proba = [0, 1]
        return proba

    # Return a distribution of recommendations for a specific user datum
    # This should a numpy array of size equal to self.n_actions, summing up to 1
    def get_action_probabilities(self, user_data):
        measurable_effect_proba = []
        for action in range(self.n_actions):
            measurable_effect_proba.append(self.predict_proba(user_data, action)[1])
        return measurable_effect_proba/np.sum(measurable_effect_proba)

    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    def recommend(self, user_data):
        cluster = self.clusters.predict(np.array([user_data]))[0]
        with np.errstate(all='ignore'):
            recommended = np.argmax(self.get_action_probabilities(user_data) + np.sqrt(2*np.log(self.t)/self.total[cluster]))
        return recommended

    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        data = np.array([user])
        cluster = self.clusters.predict(data)[0]
        self.good[cluster, action] += outcome
        self.total[cluster, action] += 1
        self.t += 1
        self.models[cluster][action].partial_fit(data, [outcome], classes=range(self.n_outcomes))
        return None

    # After all the data has been obtained, do a final analysis. This can consist of a number of things:
    # 1. Recommending a specific fixed treatment policy
    # 2. Suggesting looking at specific genes more closely
    # 3. Showing whether or not the new treatment might be better than the old, and by how much.
    # 4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
    def final_analysis(self):
        print(self.good)
        print(self.total)
        return None
