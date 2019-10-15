import pandas

## Set up for dataset
features = ['checking account balance', 'duration', 'credit history',
            'purpose', 'amount', 'savings', 'employment', 'installment',
            'marital status', 'other debtors', 'residence time',
            'property', 'age', 'other installments', 'housing', 'credits',
            'job', 'persons', 'phone', 'foreign']
target = 'repaid'
df = pandas.read_csv('../../data/credit/german.data', sep=' ',
                     names=features+[target])
import matplotlib.pyplot as plt
numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'duration', 'persons', 'credits']
quantitative_features = list(filter(lambda x: x not in numerical_features, features))
X = pandas.get_dummies(df, columns=quantitative_features, drop_first=True)
encoded_features = list(filter(lambda x: x != target, X.columns))

## Test function
def test_decision_maker(X_test, y_test, interest_rate, decision_maker):
    action = decision_maker.get_best_action(X_test)
    loss = X_test['amount']
    gain = X_test['amount']*((1 + interest_rate)**(X_test['duration']) - 1)
    utility = sum(gain[((action == 1) & (y_test == 1))]) - sum(loss[((action == 1) & (y_test == 2))])
    return utility


import nicolabk_kaiie_banker
decision_maker = nicolabk_kaiie_banker.Nicolabk_Kaiie_Banker(epsilon=10)

interest_rate = 0.005

### Do a number of preliminary tests by splitting the data in parts
def model_check(X, decision_maker, interest_rate):
  from sklearn.model_selection import train_test_split
  n_tests = 10
  utility = 0
  for iter in range(n_tests):
      print(iter, end="%\r")
      X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.2)
      decision_maker.fit(X_train, y_train)
      utility += test_decision_maker(X_test, y_test, interest_rate, decision_maker)

  print(utility / n_tests)

model_check(X, decision_maker, interest_rate)
