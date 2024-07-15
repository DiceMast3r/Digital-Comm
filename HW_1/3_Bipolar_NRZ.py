import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fft_pack

Nbits = 10         # No. bits
Nsamp = 8 + round(6/2)             # No. samples/bits
 
#1. Generate data bits a, b #
np.random.seed(20 + 6)
a = np.random.randint(0 ,2 ,Nbits)
b = 2 * a - 1

print("a = ", a)

# 2.Generate Bipolar NRZ modulated signal m(t) #
last_level = 1  # Keep track of the last voltage level for '1'
m = []
for i in range(Nbits):
    if a[i] == 1:
        m.extend([last_level] * Nsamp)
        last_level *= -1  # Alternate the level
    else:
        m.extend([0] * Nsamp)  # '0's are represented by no change (0 voltage level)

m_fft = fft_pack.fft(m)

# Adjust the FFT output and frequency axis for both positive and negative sides
m_fft_shifted = fft_pack.fftshift(m_fft)  # Shift the zero frequency to the center
f_shifted = np.linspace(-Nsamp / 2, Nsamp / 2, Nbits * Nsamp)

# Use only the second half of the FFT output for positive frequencies
m_fft_positive = m_fft[:Nbits*Nsamp // 2:]  # Take the second half
f_positive = np.linspace(0, Nsamp / 2, Nbits * Nsamp // 2) 

print("Size of a = ", a.size)
print("Size of m = ", len(m))

# 3. Create a time axis
t = np.arange(0, Nbits * Nsamp) / Nsamp

# 4. Plot the Bipolar NRZ signal
plt.figure(figsize=(10, 2))

fig_1 = plt.figure(1)
plt.step(t, m, where='post', color='red')
plt.ylim(-1.5, 1.5)
plt.title('Bipolar NRZ Modulated signal Thongchai 65010386')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)

fig_2 = plt.figure(2)
plt.plot(f_shifted, (1/Nsamp) * np.abs(m_fft_shifted), color='purple')
plt.title('Bipolar NRZ Spectrum (Both Sides) Thongchai 65010386')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

fig_3 = plt.figure(3)
plt.plot(f_positive, (1/Nsamp) * np.abs(m_fft_positive), color='purple')
plt.title('Bipolar NRZ Spectrum (Positive Side) Thongchai 65010386')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.show()