import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist


def compute_prob(x, a, b): #compute the probability of x in the range [a, b]
    P = sum((x >= a) & (x <= b)) / x.size
    return  P


N = 1000
n_normal = np.random.normal(0, 2, size=N)


print("Variance = ", np.var(n_normal))
print("Mean = ", np.mean(n_normal))

P = sum((n_normal >= -1.5) & (n_normal <= 1.5)) / N # P(-1.5 <= n_normal <= 1.5)
P_2 = sum(n_normal > 0) / N # P(n_normal < 0)

print("P(-1.5 <= n_normal <= 1.5) = ", P)
print("P(n_normal < 0) = ", P_2)
print("P(-1.5 <= n_normal <= 1.5) = ", compute_prob(n_normal, -1.5, 1.5))

plt.figure(figsize=(10, 2))
plt.plot(n_normal)
plt.ylabel('n_normal')
plt.xlabel('k')
plt.title('Normal Distribution')

plt.figure(figsize=(10, 2))
hist(n_normal, density=True, bins=40)
plt.ylabel('Probability')
plt.xlabel('n_normal')


plt.show()