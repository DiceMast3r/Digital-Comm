import numpy as np
import matplotlib.pyplot as plt

np.random.seed(50)
n1 = np.random.normal(0, 1, size=1000 * 6)

np.random.seed(666)
n2 = np.random.normal(0, 1, size=1000 * 6)


R1 = np.correlate(n1, n1, mode='full')
R1_norm = R1 / max(R1)

lag = np.arange(-n1.size + 1, n1.size)

plt.figure(figsize=(10, 2))
plt.stem(lag, R1_norm)
plt.title('Autocorrelation of n1')

R2 = np.correlate(n1, n2, mode='full')
R2_norm = R2 / max(R1)
plt.figure(figsize=(10, 2))
plt.stem(lag, R2_norm)
plt.title('Cross-correlation of n1 and n2')

r = np.corrcoef(n1, n2)
r_coef = r[0, 1]
print("Correlation coefficient = ", r_coef)

plt.show()