import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft


# Generate mixed signal
t = np.linspace(0, 1, 1000)
x = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)

# Plot mixed signal
plt.figure(figsize=(10, 4))
plt.plot(t, x)
plt.title("Mixed signal")
plt.show()

mix_fft = np.array(fft.fft(x))
freq = fft.fftfreq(len(x), 1/1000)

# Normalize the FFT
mix_fft /= len(x)

# Plot the FFT of the mixed signal
plt.figure(figsize=(10, 4))
plt.plot(freq, np.abs(mix_fft))
plt.title("FFT of mixed signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

mix_ifft = np.array(fft.ifft(mix_fft))

# Plot the IFFT of the mixed signal
plt.figure(figsize=(10, 4))
plt.plot(t, mix_ifft)
plt.title("IFFT of mixed signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

