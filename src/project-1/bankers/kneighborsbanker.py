from banker import BankerBase, run
from sklearn.neighbors import KNeighborsClassifier
import pandas
import random
from sklearn.feature_selection import SelectKBest, chi2


class KNeighborsBanker(BankerBase):
    def __init__(self, interest_rate=0.005, k=50):
        self.interest_rate = interest_rate
        self.k = k

    def set_interest_rate(self, interest_rate):
        self.interest_rate = interest_rate

    def fit(self, X, y):
        feature_selector = SelectKBest(score_func=chi2, k=10)
        feature_selector.fit(X, y)
        self.features = X.columns[feature_selector.get_support()]
        self.classifier = KNeighborsClassifier(n_neighbors=self.k)
        X = X.copy()[self.features]
        # Standardize non-categorical features
        self.mean = X[X.columns[X.dtypes == 'int64']].mean()
        self.std = X[X.columns[X.dtypes == 'int64']].std()
        X_copy = X.copy()
        X_copy.update((X[X.columns[X.dtypes == 'int64']]-self.mean)/self.std)
        self.classifier.fit(X_copy, y)

    def expected_utility(self, X):
        pr_1, pr_2 = self.predict_proba(X).T
        U_1 = X['amount']*( (1+self.interest_rate)**X['duration'] - 1)
        U_2 = -X['amount']
        return pr_1*U_1 + pr_2*U_2

    def predict_proba(self, X):
        X = X.copy()[self.features]
        # Convert series to dataframe (banker.py sends dataframes while TestLending.py sends series)
        if isinstance(X, pandas.core.series.Series):
            X_copy = X.copy().to_frame().transpose()
        else:
            X_copy = X.copy()
        # Standardize new X values
        X_copy.update((X_copy[X_copy.columns[X_copy.dtypes == 'int64']]-self.mean)/self.std)
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

if __name__ == '__main__':
    run()
