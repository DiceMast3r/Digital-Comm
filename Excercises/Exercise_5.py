import numpy as np
import matplotlib.pyplot as plot
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
plot.title('Randomly Generated Bits with Encoding')

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

# Matched filter (MF)

s_NRZL = np.array([1,1,1,1,1,1,1,1,1,1])
#s_Manchester = ....

z_t = np.convolve(r_t, s_NRZL)
#z_t = np.convolve(r,s_Manchester)
plot.figure(figsize=(10,6))
plot.plot(z_t)
plot.title('Matched Filter Output')

# A/D

z = [ ]
for i in range(Nbits):
  zz = z_t[i*Nsamp+9]
  z.append(zz)

print(np.shape(z_t))
print(np.arange(Nbits))
plot.figure(figsize=(10,6))
plot.plot(z_t)
plot.stem(np.arange(Nbits)*Nsamp+9, z, '-.')
plot.title('Sampled Signal')

# Make decision, compare z with 0
a_hat = [ ]
for zdata in z:
  if (zdata > 0):
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
