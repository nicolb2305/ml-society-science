import matplotlib.pyplot as plt
import numpy

## We want to calculate teh average salary
n_people = numpy.logspace(2, 5, 1001, dtype=int) # The number of people participating
max_salary = 10 # maximum salary of people - otherwise we can't use Laplace
epsilon = 0.5 # the amount of privacy we want to lose

# Assume the distribution of people is coming from a clipped gamma distribution. Not necessarily true
data = [numpy.random.gamma(shape=2, scale=1, size=n) for n in n_people]
for i in range(len(data)):
    for j in range(n_people[i]):
        data[i][j] = min(data[i][j], max_salary)

# Calculate the average salary
average_salary = [numpy.average(i) for i in data]
#print("The actual average salary is ", average_salary)

#### Laplace mechanism for local DP
#
# We need the sensitivity of individual data points. Since an
# individual's data can vary at most by max_salary, we have:
local_sensitivity = max_salary
# We now tune the noise to the sensitivity
local_noise = [numpy.random.laplace(scale=local_sensitivity/epsilon, size=n) for n in n_people]
# Calculate the average
local_average = [numpy.average(data[i] + local_noise[i]) for i in range(len(data))]
#print("The average salary computed with local DP + Laplace is ", local_average)

#### Laplace mechanism for centralised DP
#
# We calculate the average, so if an individual's data changes by max_salary, the average
# changes by max_salary / n. So:
central_sensitivity = [max_salary / n for n in n_people]
# We now tune sensitivity to the function
central_noise = [numpy.random.laplace(scale=i / epsilon, size=1) for i in central_sensitivity]
# Calculate the average
central_average = [numpy.average(data[i] + central_noise[i]) for i in range(len(data))]
#print("The average salary computed with central DP + Laplace is ", central_average)

plt.xscale("log")
plt.plot(n_people, average_salary)
plt.plot(n_people, local_average)
plt.plot(n_people, central_average)
plt.show()
