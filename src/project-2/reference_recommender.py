# -*- Mode: python -*-
# A simple reference recommender
#
#
# This is a medical scenario with historical data.
#
# General functions
#
# - set_reward
#
# There is a set of functions for dealing with historical data:
#
# - fit_data
# - fit_treatment_outcome
# - estimate_utiltiy
#
# There is a set of functions for online decision making
#
# - predict_proba
# - recommend
# - observe

from sklearn import linear_model
import numpy as np
from sklearn.linear_model import LogisticRegression

class HistoricalRecommender:

    #################################
    # Initialise
    #
    # Set the recommender with a default number of actions and outcomes.  This is
    # because the number of actions in historical data can be
    # different from the ones that you can take with your policy.
    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

    ## By default, the reward is just equal to the outcome, as the actions play no role.
    def _default_reward(self, action, outcome):
        return outcome

    # Set the reward function r(a, y)
    def set_reward(self, reward):
        self.reward = reward

    ##################################
    # Fit a model from patient data.
    #
    # This will generally speaking be an
    # unsupervised model. Anything from a Gaussian mixture model to a
    # neural network is a valid choice.  However, you can give special
    # meaning to different parts of the data, and use a supervised
    # model instead.
    def fit_data(self, data):
        print("Preprocessing data")
        return None


    ## Fit a model from patient data, actions and their effects
    ## Here we assume that the outcome is a direct function of data and actions
    ## This model can then be used in estimate_utility(), predict_proba() and recommend()
    def fit_treatment_outcome(self, data, actions, outcome):
        print("Fitting treatment outcomes")
        self.model = LogisticRegression(solver='sag')
        self.model.fit(data, actions)
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
        return np.sum(self.reward(actions, outcome))/len(actions)

    # Return a distribution of effects for a given person's data and a specific treatment.
    # This should be an numpy.array of length self.n_outcomes
    def predict_proba(self, data, treatment):
        return numpy.zeros(self.n_outcomes)

    # Return a distribution of recommendations for a specific user datum
    # This should a numpy array of size equal to self.n_actions, summing up to 1
    def get_action_probabilities(self, user_data):
        #print("Recommending")
        return np.ones(self.n_actions) / self.n_actions;


    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    def recommend(self, user_data):
        user_data = np.array([user_data])
        return self.model.predict(user_data)[0]

    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        return None

    # After all the data has been obtained, do a final analysis. This can consist of a number of things:
    # 1. Recommending a specific fixed treatment policy
    # 2. Suggesting looking at specific genes more closely
    # 3. Showing whether or not the new treatment might be better than the old, and by how much.
    # 4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
    def final_analysis(self):
        return None
