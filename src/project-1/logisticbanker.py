from banker import BankerBase, run
from sklearn.linear_model import LogisticRegression
import pandas
import random


class LogisticBanker(BankerBase):
    def __init__(self, interest_rate):
        self.interest_rate = interest_rate

    def fit(self, X, y):
        self.classifier = LogisticRegression(solver='newton-cg')
        self.classifier.fit(X, y)

    def expected_utility(self, X):
      pr_1, pr_2 = self.predict_proba(X).T
      U_1 = X['amount']*( (1+self.interest_rate)**X['duration'] - 1)
      U_2 = -X['amount']
      return pr_1*U_1 + pr_2*U_2

    def predict_proba(self, X):
      if isinstance(X, pandas.core.series.Series):
        return self.classifier.predict_proba(X.to_frame().transpose())
      return self.classifier.predict_proba(X)

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
