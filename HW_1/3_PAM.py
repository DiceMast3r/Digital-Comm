import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft_pack

Nbits = 10  # No. of bits (make sure it's even for 4-PAM)
Nsamp = 8 + round(6/2)  # No. of samples per symbol (not per bit)

# Generate random bits
np.random.seed(20 + 6)
a = np.random.randint(0, 2, Nbits)

# Map pairs of bits to 4-PAM symbols (-3, -1, 1, 3)
symbols = np.array([-3, -1, 1, 3])
Nsymbols = Nbits // 2  # Half as many symbols as bits
b = np.zeros(Nsymbols, dtype=int)
for i in range(0, Nbits, 2):
    # Convert pairs of bits to an index (0-3)
    idx = a[i] * 2 + a[i+1]
    b[i//2] = symbols[idx]

# Generate 4-PAM modulated signal m(t)
m = np.repeat(b, Nsamp)

# FFT of the modulated signal
m_fft = fft_pack.fft(m)
m_fft_shifted = fft_pack.fftshift(m_fft)  # Shift the zero frequency to the center
f_shifted = np.linspace(-1/(4/Nsamp), 1/(4/Nsamp), Nsymbols * Nsamp)
m_fft_positive = m_fft[:Nsymbols*Nsamp//2]
f_positive = np.linspace(0, 1/(4/Nsamp), Nsymbols * Nsamp // 2)

# Create a time axis for the 4-PAM signal
t = np.linspace(0, 2 * Nsymbols, Nsymbols * Nsamp, endpoint=False)


print("Data bit: ", a)
print("4-PAM symbols: ", b)

# Plot the 4-PAM signal with adjusted time axis
plt.figure(figsize=(10, 2))
plt.step(t, m, where='post', color='red')
plt.ylim(-4, 4)
plt.title('4-PAM Modulated Signal Thongchai 65010386')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot the 4-PAM Spectrum showing both positive and negative sides with adjusted frequency axis
plt.figure(figsize=(10, 4))
plt.plot(f_shifted, (1/Nsamp) * np.abs(m_fft_shifted), color='purple')
plt.title('4-PAM Spectrum (Both Sides) Thongchai 65010386')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

# Plot the spectrum of the 4-PAM signal (positive side only) with adjusted frequency axis
plt.figure(figsize=(10, 4))
plt.plot(f_positive, (1/Nsamp) * np.abs(m_fft_positive), color='purple')
plt.title('4-PAM Spectrum (Positive Side) Thongchai 65010386')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.show()