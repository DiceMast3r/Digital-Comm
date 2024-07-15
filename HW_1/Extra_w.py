from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter
from scipy.fft import fft, fftfreq
import scipy.fftpack as fft_pack
from scipy.signal import freqz



# Raised Cosine Filter Coefficients
def raised_cosine(t, beta, sps):
    # Handle the special case when beta = 0
    if beta == 0:
        return np.sinc(t)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))
        denominator = np.pi * t * (1 - (4 * beta * t) ** 2)
        response = numerator / denominator
        response = np.where(np.abs(t) == 1 / (4 * beta), beta / np.sqrt(2) * ((1 + np.sin(np.pi / (4 * beta))) + (1 - np.sin(np.pi / (4 * beta)))), response)
        response = np.where(t == 0, 1.0, response)
    return response


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

# Raised Cosine Filter Parameters
beta = 0  # Roll-off factor
N = 6  # Number of symbols
sps = 10  # Samples per symbol

# Time vector for the filter
t_f = np.arange(-N/2, N/2, 1/sps)

filter_coeffs = raised_cosine(t_f, beta, sps)

# Apply the filter to the modulated signal
filtered_signal = np.convolve(m, filter_coeffs, mode='same')

# Calculate the spectrum of the filtered signal
f, H = freqz(filtered_signal)


print("Data bit: ", a)
print("4-PAM symbols: ", b)

# Create a time axis for the 4-PAM signal
t = np.linspace(0, 2 * Nsymbols, Nsymbols * Nsamp, endpoint=False)

# Plot the 4-PAM signal with adjusted time axis
plt.figure(figsize=(10, 2))
plt.step(t, m, where='post', color='red')
plt.ylim(-4, 4)
plt.title('4-PAM Modulated Signal Thongchai 65010386')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot the spectrum of the 4-PAM signal (positive side only) with adjusted frequency axis
plt.figure(figsize=(10, 4))
plt.plot(f_positive, (1/Nsamp) * np.abs(m_fft_positive), color='purple')
plt.title('4-PAM Spectrum (Positive Side) Thongchai 65010386')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

# Plot the spectrum
plt.figure(figsize=(10, 6))
plt.plot(f, (1 / (Nsamp * 10)) * np.abs(H), color='purple')
plt.title('Spectrum of the Filtered Signal (beta = {0})'.format(beta))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.show()