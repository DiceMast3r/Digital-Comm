import numpy as np
import matplotlib.pyplot as plt
import math

Nbits = 1000
Nsamp = 20

# 1. Generate data bits
np.random.seed(10)
a = np.random.randint(0, 2, Nbits)
#print(a)

f1 = 4
f2 = 6
t = np.arange(0, 1, 1/(f1*Nsamp))
cos_t_f1 = np.cos(2*np.pi*f1*t)
cos_t_f2 = np.cos(2*np.pi*f2*t)

# Modulate
x_t = []
for i in range(Nbits):
    if a[i] == 0:
        x_t.extend(cos_t_f1)
    else:
        x_t.extend(cos_t_f2)

tt = np.arange(0, Nbits, 1/(f1*Nsamp))

"""plt.figure(figsize=(10, 5))
plt.plot(tt, x_t)
plt.title("BFSK Modulated signal x(t)")"""

#  Generate Gaussian noise
mu = 0
sigma = 1

n_t = np.random.normal(mu, sigma, np.size(x_t) )
print(np.var(n_t))
"""plt.figure(figsize=(10, 5))
plt.plot(n_t)
plt.title("Gaussian noise")"""

# received signal
r_t = x_t +n_t

"""plt.figure(figsize=(10, 5))
plt.plot(r_t)
plt.title("Received signal r(t)")"""

# Correlator

z = [ ]
z_tt = [ ]
for i in range(Nbits):
  z_t = np.multiply(r_t[i*f1*Nsamp:(i+1)*f1*Nsamp], cos_t_f1)
  z_t_2 = np.multiply(r_t[i*f1*Nsamp:(i+1)*f1*Nsamp], cos_t_f2)
  z_t = z_t/(f1*Nsamp*0.5)     # r(t)xcos of each bit period
  z_t_2 = z_t_2/(f2*Nsamp*0.5)
  z_t_out = sum(z_t)          # output of summation/correlato at each bit period
  z_t_out_2 = sum(z_t_2)
  z_tt.extend(z_t)            # r(t)xcos at all time
  z_tt.extend(z_t_2)
  z.append(z_t_out_2 - z_t_out)           # output of correlator at all time

"""plt.figure(figsize=(10, 5))
plt.plot(z_tt)
plt.title("Correlator output z(t)")

plt.figure(figsize=(10, 5))
plt.stem(z)
plt.title("Correlator output z[k]")"""

#plot signal vectors, constellation of correlator output z
"""plt.figure(figsize=(10, 5))
plt.scatter(z, np.zeros(Nbits), color='b')
plt.scatter([1,0],[0, 0], color='r')
plt.title("Constellation of correlator output z")"""

plt.figure(figsize=(10, 5))
plt.hist(z, density=True, bins=100)
plt.title("Histogram of correlator output z (Sigma = {0})".format(sigma))

# Make decision, compare z with threshold 
a_hat = []
threshold = 0
for zz_1 in z:
    if zz_1 > threshold:
        a_hat.append(1)
    else:
        a_hat.append(0)
        
# Calculate the bit error rate
err_num = sum((a != a_hat))
print('err_num = ', err_num)

ber = err_num / Nbits
print('BER = ', ber)

plt.show()