import numpy as np
import matplotlib.pyplot as plot
import math as math

Nbits = 15
Nsamp = 10
np.random.seed(30)
a = np.random.randint(0, 2, Nbits)

# Generate Unipolar signals
x_t = []
for i in range(Nbits):
    if a[i] == 1:
        x_t.extend([1] * Nsamp)
    else:
        x_t.extend([0] * Nsamp)

plot.figure(figsize=(10, 6))
plot.plot(x_t)
plot.title('Unipolar Signal')

# Generate AWGN
mu = 0
sigma = 0.72
n_t = np.random.normal(mu, sigma, Nbits * Nsamp)

# Received signals
r_t = x_t + n_t
plot.figure(figsize=(10, 6))
plot.plot(r_t)
plot.title('Received Signal')

# Matched filters (MF) for Unipolar coding
s_Unipolar_1 = np.array([1] * Nsamp)  # Matched filter for '1'
s_Unipolar_2 = np.array([0] * Nsamp)  # Matched filter for '0'

z_t_1 = np.convolve(r_t, s_Unipolar_1)
z_t_2 = np.convolve(r_t, s_Unipolar_2)

"""plot.figure(figsize=(10, 6))
plot.plot(z_t_1)
plot.title('Matched Filter 1 Output')

plot.figure(figsize=(10, 6))
plot.plot(z_t_0)
plot.title('Matched Filter 2 Output')"""

# A/D
z_1 = []
z_2 = []
for i in range(Nbits):
    zz_1 = z_t_1[i * Nsamp + Nsamp - 1]
    zz_2 = z_t_2[i * Nsamp + Nsamp - 1]
    z_1.append(zz_1)
    z_2.append(zz_2)

plot.figure(figsize=(10, 6))
plot.plot(z_t_1)
plot.stem(np.arange(Nbits) * Nsamp + Nsamp - 1, z_1, '-.')
plot.title('Sampled Signal from Matched Filter 1')

plot.figure(figsize=(10, 6))
plot.plot(z_t_2)
plot.stem(np.arange(Nbits) * Nsamp + Nsamp - 1, z_2, '-.')
plot.title('Sampled Signal from Matched Filter 2')

# Make decision, compare z with threshold (Nsamp / 2)
a_hat = []
threshold = Nsamp / 2
for zz_1, zz_2 in zip(z_1, z_2):
    if zz_1 > threshold:
        a_hat.append(1)
    else:
        a_hat.append(0)

# Calculate the bit error rate
err_num = sum((a != a_hat))
print('err_num = ', err_num)

# Calculate Eb/N0
Eb = np.mean(np.array(x_t) ** 2)
N0 = 2 * (sigma ** 2)
EbN0 = 10 * math.log10(Eb / N0)
print('Eb/N0 = {0} dB'.format(EbN0))

plot.figure(figsize=(10, 6))
plot.stem(a_hat)
plot.title('Decoded Signal')

plot.show()