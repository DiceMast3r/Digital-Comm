import numpy as np
import matplotlib.pyplot as plot
import math as math

Nbits = 16
Nsamp = 10
np.random.seed(30)
a = np.random.randint(0,2,Nbits)
b = 2 * a - 1

'''plot.figure(figsize=(10,6))
plot.stem(a)
plot.title('Randomly Generated Bits')

plot.figure(figsize=(10,6))
plot.stem(b)
plot.title('Randomly Generated Bits with Encoding')'''

# Generate Manchester encoded signal m(t) #
x_t = []
for i in range(Nbits):
    if a[i] == 1:
        x_t.extend([1] * (Nsamp // 2) + [-1] * (Nsamp // 2))
    else:
        x_t.extend([-1] * (Nsamp // 2) + [1] * (Nsamp // 2))

'''plot.figure(figsize=(10,6))
plot.plot(x_t)
plot.title('NRZ-L Modulated Signal')'''

# Generate AWGN
mu = 0
sigma = 0.65
n_t = np.random.normal(mu, sigma, Nbits*Nsamp )
print(np.shape(n_t))
'''plot.figure(figsize=(10,6))
plot.plot(n_t)
plot.title('AWGN Signal')'''

# received signals
r_t = x_t + n_t
'''plot.figure(figsize=(10,6))
plot.plot(r_t)
plot.title('Received Signal')'''

# Correlator
s_NRZL = np.array([1]*Nsamp) #for NRZ-L
s_Manchester = np.array([1,1,1,1,1,-1,-1,-1,-1,1]) #for Manchester
z = []
for i in range(Nbits):
    z_t = np.multiply(r_t[i*Nsamp:(i+1)*Nsamp], s_Manchester)
    z_t_out = sum(z_t)
    z.append(z_t_out)
    
plot.figure(figsize=(10,6))
plot.stem(z)
plot.title('Correlator Output')

plot.figure(figsize=(10,6))
plot.hist(z, density=True, bins=20)
plot.title('Histogram of Correlator Output')

# Make decision, compare z with 0
a_hat = [ ]
for zdata in z:
  if (zdata > 0):
    a_hat.append(1)
  else:
    a_hat.append(0)
    
'''plot.figure(figsize=(10,6))
plot.stem(a_hat)
plot.title('Decoded Signal')'''

# Calculate the bit error rate
err_num = sum((a!=a_hat))
print('err_num = ', err_num)

# Calculate Eb/N0
Eb = 10
N0 = 2 * (sigma**2)
EbN0 = 10 * math.log10(Eb/N0)
print('SNR (dB) = {0:.5f} dB'.format(EbN0))

plot.show()