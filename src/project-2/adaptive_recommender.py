import numpy as np
from random_recommender import RandomRecommender

class AdaptiveRecommender(RandomRecommender):
    def __init__(self, n_actions, n_outcomes, alpha=0.1, d_features=130):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

        # Initialize variables for LinUCB
        self.alpha = alpha
        self.d_features = d_features
        self.A = np.zeros((self.n_actions, self.d_features, self.d_features))
        self.b = np.zeros((self.n_actions, self.d_features))
        for i in range(self.n_actions):
            self.A[i] = np.identity(self.d_features)

    # Just calls observe to fit our ridge regression for the actions in the
    # training data
    def fit_treatment_outcome(self, data=None, actions=None, outcome=None):
        self.fit_data(data)
        print("Fitting treatment outcomes")
        self.actions_taken = np.zeros((self.n_actions, 2), dtype=int)
        if not (data is None and actions is None and outcome is None):
            for i in range(len(data)):
                self.observe(data[i], actions[i][0], outcome[i][0], count=False)
        return None

    # Calculates probabilities of outcomes using the given treatment's ridge
    # regression
    def predict_proba(self, data, treatment):
        theta_hat_a = np.matmul(np.linalg.inv(self.A[treatment]), self.b[treatment])
        p_t_a = np.matmul(data.T, theta_hat_a)
        return np.array([1-p_t_a, p_t_a])

    # LinUCB implementation using ridge regression
    def get_action_probabilities(self, user_data):
        theta_hat_a = np.zeros((self.n_actions, self.d_features))
        p_t_a = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            Q = np.linalg.inv(self.A[a])
            theta_hat_a[a] = np.matmul(Q, self.b[a])
            # Set expected payoff from treatment a to 0 if it's a gene silencing
            # treatment and the patient does not have the gene
            if self.n_actions > 2 and user_data[a-1] == 0:
                p_t_a[a] = 0
            else:
                p_t_a[a] = np.matmul(user_data.T, theta_hat_a[a]) + self.alpha*np.sqrt(np.matmul(np.matmul(user_data.T, Q), user_data))
        action_probabilities = np.zeros(self.n_actions)
        actions = np.argwhere(p_t_a == np.amax(p_t_a))
        action_probabilities[actions] = 1/actions.size
        return action_probabilities

    # Updating LinUCB's ridge regression
    # Used for fititng as  well when count=False
    def observe(self, user, action, outcome, count=True):
        self.A[action] += np.outer(user, user)
        self.b[action] += self.reward(action, outcome)*user

        if count:
            self.actions_taken[action][1] += outcome
            self.actions_taken[action][0] += 1
        return None

    # After all the data has been obtained, do a final analysis. This can consist of a number of things:
    # 1. Recommending a specific fixed treatment policy
    # 2. Suggesting looking at specific genes more closely
    # 3. Showing whether or not the new treatment might be better than the old, and by how much.
    # 4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
    def final_analysis(self):
        ratio = self.actions_taken[:, 1]/self.actions_taken[:, 0]
        action_string_length = len(str(self.n_actions-1))
        print(f"{'Given':>{17+action_string_length}}  {'Cured'}  {'Ratio'}  {'Features x with 5 largest coeffs'}")
        important_features = np.zeros(self.d_features, dtype=int)
        for a in range(self.n_actions):
            theta_hat_a = np.matmul(np.linalg.inv(self.A[a]), self.b[a])
            highest_coeffs = np.argsort(np.abs(theta_hat_a))[:5]
            important_features[highest_coeffs] += 1
            highest_coeffs_string = f"{highest_coeffs[0]+1:>3} {highest_coeffs[1]+1:>3} {highest_coeffs[2]+1:>3} {highest_coeffs[3]+1:>3} {highest_coeffs[4]+1:>3}"
            print(f"Treatment {a:>{action_string_length}}: {self.actions_taken[a][0]:>5}  {self.actions_taken[a][1]:>5}  {ratio[a]:>5.2f}  {highest_coeffs_string:>32}")
        print("\n", end="")

        if self.n_actions > 2:
            for a in range(self.n_actions):
                if a > 2 and ratio[a] > 0:
                    print(f"Gene silencing treatment for gene {f'x{a}':>4} had a success rate of {ratio[a]:.2f}.")
            print("\n", end="")

            print("Number of times a gene was among the 5 largest for an action, suggest looking at these genes more closely:")
            for a in np.argsort(important_features)[::-1]:
                if important_features[a] < 5: break
                if a > 2 and a < 128:
                    print(f"Gene {f'x{a}':>4}: {important_features[a]}")
            print("\n", end="")

            if ratio[1] > ratio[2]:
                print(f"Using our models the old treatment 1 is better than the new treatment 2 with curing rations of {ratio[1]:.2f} and {ratio[2]:.2f}.")
            else:
                print(f"Using our models the old treatment 2 is better than the new treatment 1 with curing rations of {ratio[2]:.2f} and {ratio[1]:.2f}.")
        return None
