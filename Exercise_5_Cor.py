from matplotlib import figure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
import math as math

Nbits = 15
Nsamp = 10
np.random.seed(30)
a = np.random.randint(0,2,Nbits)
b = 2*a - 1
plot.figure(figsize=(10,6))
plot.stem(a)
plot.title('Randomly Generated Bits')

plot.figure(figsize=(10,6))
plot.stem(b)
plot.title('Randomly Generated Bits with NRZ-L Encoding')

# Generate NRZ-L Modulated signals
x_t=[]
for i in range(Nbits):
  if(a[i]==1):
    x_t.extend([1]*Nsamp)
  else:
    x_t.extend([-1]*Nsamp)

plot.figure(figsize=(10,6))
plot.plot(x_t)
plot.title('NRZ-L Modulated Signal')

# Generate AWGN
mu = 0
sigma = 0.2
n_t = np.random.normal(mu, sigma, Nbits*Nsamp )
print(np.shape(n_t))
plot.figure(figsize=(10,6))
plot.plot(n_t)
plot.title('AWGN Signal')

# received signals
r_t = x_t + n_t
plot.figure(figsize=(10,6))
plot.plot(r_t)
plot.title('Received Signal')

# Correlator
s_NRZL = np.array([1]*Nsamp)
correlation = np.correlate(r_t, s_NRZL, mode='full')
plot.figure(figsize=(10,6))
plot.plot(correlation)
plot.title('Correlator Output')

# A/D
z = [ ]
for i in range(Nbits):
  zz = correlation[i*Nsamp + Nsamp - 1]
  z.append(zz)

print(np.shape(correlation))
print(np.arange(Nbits))
plot.figure(figsize=(10,6))
plot.stem(np.arange(Nbits)*Nsamp + Nsamp - 1, z, '-.')
plot.title('Sampled Signal')

# Make decision, compare z with 0
a_hat = [ ]
for zdata in z:
  if (zdata > 0):
    a_hat.append(1)
  else:
    a_hat.append(0)

# compute error numbers
err_num = sum((a!=a_hat))
print('err_num = ', err_num)
plot.figure(figsize=(10,6))
plot.stem(a_hat)
plot.title('Decoded Signal')

plot.show()