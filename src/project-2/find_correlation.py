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

#print(pandas.crosstab(X[:, 5], X[:, 128]))

#Benjamini, Y.&Hochberg, Y.(1995). Controlling the false discovery rate:a practical and powerful approach to multiple testing.Journal of the RoyalStatistical Society. Series B (Methodological), 289–300.

sorted_features = np.argsort(ps, axis=1)
M = 127
q = 0.05 # FDR, False Discovery Rate
significant_features = [[], []]

for i in range(2):
    for j in range(128):
        if ps[i][sorted_features[i][j]] < q*(j+1)/M:
            significant_features[i].append(sorted_features[i][j])
        else:
            break

insignificant_features = [
    [x for x in range(1, 129) if x not in significant_features[0]],
    [x for x in range(1, 129) if x not in significant_features[1]]
]

print(f"Significant features for X_129 with false discovery rate {q}:\n{sorted(significant_features[0])}\n")
print(f"Non-significant features for X_129 with false discovery rate {q}:\n{sorted(insignificant_features[0])}\n")
print(f"Significant features for X_130 with false discovery rate {q}:\n{sorted(significant_features[1])}\n")
print(f"Non-significant features for X_129 with false discovery rate {q}:\n{sorted(insignificant_features[1])}")
