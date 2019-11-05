import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

X = pandas.read_csv('data/medical/historical_X.dat', header=None, sep=" ").values

chi2s = []
ps = []
for i in [128, 129]:
    chi2s.append([])
    ps.append([])
    for j in range(128):
        contingency = pandas.crosstab(X[:, j], X[:, i]).values
        chi2, p, dof, ex = chi2_contingency(contingency)
        # print(f'X_{j}, X_{i}: chi2={chi2:10.3f}, p={p:.5e}')
        chi2s[-1].append(chi2)
        ps[-1].append(p)
        print(f'{(100*j)//256 + (0 if i == 128 else 50)}%', end='\r')

width=0.35
plt.bar(np.arange(1, 129)-width/2, chi2s[0], width = width, label=r'$x_{129}$')
plt.bar(np.arange(1, 129)+width/2, chi2s[1], width = width, label=r'$x_{130}$')
plt.xticks(range(1, 129, 8), [fr'$x_{"{"}{i}{"}"}$' for i in range(1, 129, 8)])
plt.title(r'$\chi^2$-test statistic between $x_1:x_{128}$ and $x_{129}, x_{130}$')
plt.legend()
plt.savefig('chi2_statistic.png')

for i in range(2):
    for j in range(128):
        if chi2s[i][j] >= 1000:
            print(f'{f"X{j+1} is important for symptom X{129+i}":>34s}')
