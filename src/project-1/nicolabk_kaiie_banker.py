from banker import BankerBase, run
from sklearn.ensemble import RandomForestClassifier
import pandas
import random
from sklearn.model_selection import cross_val_score
import numpy as np


class Nicolabk_Kaiie_Banker(BankerBase):
    def __init__(self, interest_rate=0.005):
        self.interest_rate = interest_rate
        self.best_n = None

    def set_interest_rate(self, interest_rate):
        self.interest_rate = interest_rate

    def fit(self, X, y):
        if not self.best_n:
            n_estimators = range(50, 125, 25)
            clfs = [RandomForestClassifier(n_estimators=n) for n in n_estimators]
            scores = [np.mean(cross_val_score(clf, X, y, cv=40, scoring=self.utility_scoring, n_jobs=-1)) for clf in clfs]
            self.best_n = n_estimators[np.argmax(np.array(scores))]
            print(f"n_estimators: {self.best_n}")
        self.classifier = RandomForestClassifier(n_estimators=self.best_n, n_jobs=-1)
        self.classifier.fit(X, y)

    def utility_scoring(self, estimator, X, y):
        estimator.fit(X, y)
        pr_1, pr_2 = estimator.predict_proba(X).T
        U_1 = X['amount']*( (1+self.interest_rate)**X['duration'] - 1)
        U_2 = -X['amount']
        return sum(pr_1*U_1 + pr_2*U_2)

    def expected_utility(self, X):
        pr_1, pr_2 = self.predict_proba(X).T
        U_1 = X['amount']*( (1+self.interest_rate)**X['duration'] - 1)
        U_2 = -X['amount']
        return pr_1*U_1 + pr_2*U_2

    def predict_proba(self, X):
        # Convert series to dataframe (banker.py sends dataframes while TestLending.py sends series)
        if isinstance(X, pandas.core.series.Series):
            X_copy = X.copy().to_frame().transpose()
        else:
            X_copy = X.copy()
        return self.classifier.predict_proba(X_copy)

    def get_best_action(self, X):
        if isinstance(X, pandas.core.series.Series):
            for exp_utility in self.expected_utility(X):
                if exp_utility > 0:
                    return 1
                else:
                    return 2
        else:
            actions = []
            for exp_utility in self.expected_utility(X):
                if exp_utility > 0:
                    actions.append(1)
                else:
                    actions.append(2)
        return pandas.Series(actions, index = X.index)

    def get_diff_privacy(self, y, epsilon=0.1):
        central_sensitivity = 1/len(y)
        central_noise = np.random.laplace(scale=central_sensitivity/epsilon)
        return np.bincount(y)[1] / len(y) + central_noise

if __name__ == '__main__':
    run()
