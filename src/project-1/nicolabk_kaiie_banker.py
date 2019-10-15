from banker import BankerBase, run
from sklearn.ensemble import RandomForestClassifier
import pandas
import random
from sklearn.model_selection import cross_val_score
import numpy as np

#pandas.set_option('display.max_columns', None)
pandas.set_option('mode.chained_assignment', None)

class Nicolabk_Kaiie_Banker(BankerBase):
    def __init__(self, interest_rate=0.005, epsilon=None):
        self.interest_rate = interest_rate
        if epsilon:
            self.epsilon = epsilon/20
        else:
            self.epsilon = epsilon

    def set_interest_rate(self, interest_rate):
        self.interest_rate = interest_rate

    def fit(self, X, y):
        n_estimators = range(50, 125, 25)
        clfs = [RandomForestClassifier(n_estimators=n) for n in n_estimators]
        scores = [np.mean(cross_val_score(clf, X, y, cv=10, scoring=self._utility_scoring, n_jobs=-1)) for clf in clfs]
        best_n = n_estimators[np.argmax(np.array(scores))]
        self.classifier = RandomForestClassifier(n_estimators=best_n, n_jobs=-1)
        if self.epsilon:
            self.sensitivity = {}
            for column in X.columns[X.dtypes == 'int64']:
                self.sensitivity[column] = X[column].max()-X[column].min()
        self.classifier.fit(X, y)

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

        if self.epsilon:
            X_copy = self._privacy(X_copy)
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

    def _utility_scoring(self, estimator, X, y):
        estimator.fit(X, y)
        pr_1, pr_2 = estimator.predict_proba(X).T
        U_1 = X['amount']*( (1+self.interest_rate)**X['duration'] - 1)
        U_2 = -X['amount']
        return sum(pr_1*U_1 + pr_2*U_2)

    def _privacy(self, X):
        X_private = X.copy()
        for column in self.sensitivity:
            self._laplace_mechanism(X_private[column], self.sensitivity[column])
        cat_col = sorted(list(set(X.columns) - set(self.sensitivity.keys())))
        cat_grouped = []
        prev = ""
        for cat in cat_col:
            split = cat.split('_')[0]
            if split == prev:
                cat_grouped[-1].append(cat)
            else:
                cat_grouped.append([cat])
            prev = split

        for cat in cat_grouped:
            self._exponential(X_private, cat)
        return X_private


    def _exponential(self, X, category):
        quality = np.zeros(len(category)+1)
        for i in range(len(category)):
            if X[category[i]].iloc[0] == 1:
                quality[i+1] = 1
                X[category[i]].iloc[0] = 0
                break

        if np.count_nonzero(quality) == 0:
            quality[0] = 1

        pr_cat = np.exp(self.epsilon*quality)/sum(np.exp(self.epsilon*quality))
        choice = np.random.choice(['filler'] + category, p=pr_cat)
        if choice != 'filler':
            X[choice] = 1

    def _laplace_mechanism(self, X, sensitivity):
        noise = np.random.laplace(scale=sensitivity/self.epsilon, size=X.size)
        X += noise

if __name__ == '__main__':
    run()
