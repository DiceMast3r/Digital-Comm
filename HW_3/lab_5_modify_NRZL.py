import numpy as np
import matplotlib.pyplot as plot
import math as math

Nbits = 15
Nsamp = 10
np.random.seed(30)
a = np.random.randint(0,2,Nbits)
b = 2*a - 1

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

# received signals
r_t = x_t + n_t

# Matched filters (MF)
s_NRZL_1 = np.array([1]*Nsamp)
s_NRZL_2 = np.array([-1]*Nsamp)

z_t_1 = np.convolve(r_t, s_NRZL_1)
z_t_2 = np.convolve(r_t, s_NRZL_2)
"""
plot.figure(figsize=(10,6))
plot.plot(z_t_1)
plot.title('Matched Filter 1 Output')

plot.figure(figsize=(10,6))
plot.plot(z_t_2)
plot.title('Matched Filter 2 Output')"""

# A/D
z_1 = [ ]
z_2 = [ ]
for i in range(Nbits):
  zz_1 = z_t_1[i*Nsamp+Nsamp-1]
  zz_2 = z_t_2[i*Nsamp+Nsamp-1]
  z_1.append(zz_1)
  z_2.append(zz_2)

print(np.shape(z_t_1))
print(np.arange(Nbits))
plot.figure(figsize=(10,6))
plot.plot(z_t_1)
plot.stem(np.arange(Nbits)*Nsamp+Nsamp-1, z_1, '-.')
plot.title('Sampled Signal from Matched Filter 1')

plot.figure(figsize=(10,6))
plot.plot(z_t_2)
plot.stem(np.arange(Nbits)*Nsamp+Nsamp-1, z_2, '-.')
plot.title('Sampled Signal from Matched Filter 2')

# Make decision, compare z with 0
a_hat = [ ]
for zz_1, zz_2 in zip(z_1, z_2):
  if (zz_1 > zz_2):
    a_hat.append(1)
  else:
    a_hat.append(0)

# Calculate the bit error rate
err_num = sum((a!=a_hat))
print('err_num = ', err_num)

# Calculate Eb/N0
Eb = np.mean(b**2)
N0 = 2 * (sigma**2)
EbN0 = 10*math.log10(Eb/N0)
print('Eb/N0 = {0} dB'.format(EbN0))

plot.figure(figsize=(10,6))
plot.stem(a_hat)
plot.title('Decoded Signal')

plot.show()