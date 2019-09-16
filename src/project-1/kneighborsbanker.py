from banker import BankerBase, run
from sklearn.neighbors import KNeighborsClassifier
import pandas
import random
from sklearn import preprocessing


class KNeighborsBanker(BankerBase):
    def __init__(self, interest_rate, k=15):
        self.interest_rate = interest_rate
        self.k = k

    def fit(self, X, y):
        self.classifier = KNeighborsClassifier(n_neighbors=self.k)
        self.max = X.iloc[:, -7:-1].max()
        self.min = X.iloc[:, -7:-1].min()
        X_copy = X.copy()
        X_copy.update((X.iloc[:, -7:-1]-self.min)/(self.max-self.min))
        """
        self.scaler = preprocessing.Normalizer().fit(X.iloc[:, -7:-1])
        X.loc[:, -7:-1] = self.scaler.transform(X.iloc[:, -7:-1])
        print(X.head())
        """
        self.classifier.fit(X, y)

    def get_expected_utility(self, X):
      pr_1, pr_2 = self.predict_proba(X).T
      U_1 = X['amount']*( (1+self.interest_rate)**X['duration'] - 1)
      U_2 = -X['amount']
      #print(f"{pr_1} {pr_2} {U_1} {U_2} {X['duration']} {pr_1*U_1 + pr_2*U_2}")
      return pr_1*U_1 + pr_2*U_2

    def get_proba(self):
      return random.random()

    def predict_proba(self, X):
      X.update((X.iloc[:, -7:-1]-self.min)/(self.max-self.min))
      if isinstance(X, pandas.core.series.Series):
        return self.classifier.predict_proba(X.to_frame().transpose())
      return self.classifier.predict_proba(X)

    def get_best_action(self, X):
      if isinstance(X, pandas.core.series.Series):
        for exp_utility in self.get_expected_utility(X):
          if exp_utility > 0:
            return 1
          else:
            return 2
      else:
        actions = []
        for exp_utility in self.get_expected_utility(X):
          if exp_utility > 0:
            actions.append(1)
          else:
            actions.append(2)
        return pandas.Series(actions, index = X.index)

if __name__ == '__main__':
    run()
