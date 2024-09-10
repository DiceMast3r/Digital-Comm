import array
import numpy as np
import matplotlib.pyplot as plot
import komm
import scipy.fft as fft

# Generate time axis
fs = 5000  # Sample rate (Hz)
t = np.linspace(0, 1, fs)

# Generate cosine signals
f1 = 50  # Frequency of first cosine (5 kHz)
f2 = 100  # Frequency of second cosine (10 kHz)
x1 = np.cos(2 * np.pi * f1 * t)  # First cosine signal
x2 = np.cos(2 * np.pi * f2 * t)  # Second cosine signal

x1_fft = np.array(fft.fft(x1))  # Compute the FFT of the first cosine signal
x2_fft = fft.fft(x2)  # Compute the FFT of the second cosine signal

# Plot the magnitude spectrum of the first cosine signal
plot.figure(figsize=(10, 5))
plot.plot(np.abs(x1_fft))
plot.title('Magnitude Spectrum of 5 kHz Cosine Signal')
plot.xlabel('Frequency (Hz)')
plot.ylabel('Magnitude')
plot.grid(True)
plot.show()

# Plot the signals
plot.plot(t, x1, label='5 kHz')
plot.plot(t, x2, label='10 kHz')
plot.xlabel('Time (s)')
plot.ylabel('Amplitude')
plot.title('Cosine Signals at 5 kHz and 10 kHz')
plot.legend()
plot.grid(True)
plot.show()