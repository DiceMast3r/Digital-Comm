import numpy as np
import matplotlib.pyplot as plot
import math as math
import time 

# start timer
start_time = time.time()

Nbits = 150000
Nsamp = 10
np.random.seed(30)
a = np.random.randint(0,2,Nbits)
b = 2 * a - 1
Eb = 10

plot.figure(figsize=(10,6))
plot.stem(a)
plot.title('Transmitted Data')

mode = input('Enter the mode (Manchester(M) , NRZ-L(N), Unipolar RZ (U), Polar RZ (P)): ')

# Generate NRZ-L or Manchester Modulated signals
x_t = []
if mode == 'N':
    for i in range(Nbits):
        if a[i] == 1:
            x_t.extend([1] * Nsamp)
        else:
            x_t.extend([-1] * Nsamp)
elif mode == 'M':
    for i in range(Nbits):
        if a[i] == 1:
            x_t.extend([1] * (Nsamp // 2) + [-1] * (Nsamp // 2))
        else:
            x_t.extend([-1] * (Nsamp // 2) + [1] * (Nsamp // 2))
elif mode == 'U':
    for i in range(Nbits):
        if a[i] == 1:
            x_t.extend([1] * (Nsamp // 2) + [0] * (Nsamp // 2))
        else:
            x_t.extend([0] * (Nsamp // 2) + [0] * (Nsamp // 2))
elif mode == 'P':
    for i in range(Nbits):
        if a[i] == 1:
            x_t.extend([1] * (Nsamp // 2) + [0] * (Nsamp // 2))
        else:
            x_t.extend([-1] * (Nsamp // 2) + [0] * (Nsamp // 2))
else:
    print('Invalid mode')
    exit()

plot.figure(figsize=(10,6))
plot.plot(x_t)
plot.grid(True)
plot.title('Transmitted Signal (Mode: {0})'.format(mode))

# Generate AWGN
mu = 0
sigma = 0.707106
N0 = 2 * (sigma**2)

snr_m = 10 * math.log10(Eb/N0)
n_t = np.random.normal(mu, sigma, Nbits*Nsamp )

# received signals
r_t = x_t + n_t

# Correlator
s_NRZL = np.array([1] * Nsamp) #for NRZ-L
s_Manchester = np.array([1,1,1,1,1,-1,-1,-1,-1,-1]) #for Manchester
s_PRZ = np.array([1,0,1,0,1,0,1,0,1,0]) #for Polar RZ
s_URZ = np.array([1,0,1,1,1,1,1,0,0,0]) #for Unipolar NRZ
z = []
if mode == 'N':
    for i in range(Nbits):
        z_t = np.multiply(r_t[i * Nsamp:(i+1) * Nsamp], s_NRZL)
        z_t_out = sum(z_t)
        z.append(z_t_out)
elif mode == 'M':
    for i in range(Nbits):
        z_t = np.multiply(r_t[i * Nsamp:(i+1) * Nsamp], s_Manchester)
        z_t_out = sum(z_t)
        z.append(z_t_out)
elif mode == 'U':
    for i in range(Nbits):
        z_t = np.multiply(r_t[i * Nsamp:(i+1) * Nsamp], s_URZ)
        z_t_out = sum(z_t)
        z.append(z_t_out)
elif mode == 'P':
    for i in range(Nbits):
        z_t = np.multiply(r_t[i * Nsamp:(i+1) * Nsamp], s_PRZ)
        z_t_out = sum(z_t)
        z.append(z_t_out)
else:
    print('Invalid mode')
    exit()
    
print("Mode: " + mode)
plot.figure(figsize=(10,6))
plot.stem(z)
plot.title('Correlator Output (Mode: {0})'.format(mode))

plot.figure(figsize=(10,6))
plot.hist(z, density=True, bins=20)
plot.title('Histogram of Correlator Output (SNR = {0:.3f} dB) Mode: {1}'.format(snr_m, mode))

# Make decision, compare z with 0
a_hat = [ ]
for zdata in z:
  if (zdata > 0):
    a_hat.append(1)
  else:
    a_hat.append(0)


plot.figure(figsize=(10,6))
plot.stem(a_hat)
plot.title('Received Data (Mode: {0})'.format(mode))

# Calculate error
err_num = sum((a!=a_hat))
print('err_num = ', err_num)

# Calculate BER
BER = err_num/Nbits
print('BER = {0:.3f}'.format(BER))

# Calculate Eb/N0
EbN0 = 10 * math.log10(Eb/N0)
print('SNR (dB) = {0:.3f} dB'.format(EbN0))

# End the timer
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print("Program execution time for {0} bits: {1:.3f} seconds".format(Nbits, elapsed_time))

#plot.show()