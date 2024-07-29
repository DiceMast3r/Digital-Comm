import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, welch
from scipy.fft import fft, fftfreq, fftshift

# Define the Gaussian signal x(t)
def gaussian_signal(t, mean=0, variance=1):
    return np.random.normal(mean, np.sqrt(variance), len(t))

# Define the rectangular filter h(t) = rect(t/16)
def rect(t, T=16):
    return np.where(np.abs(t) <= T/2, 1, 0)

# Time vector
t = np.linspace(-32, 32, 1024)

# Generate the Gaussian signal
x_t = gaussian_signal(t)

# Generate the rectangular filter response
h_t = rect(t)

# Convolve x(t) with h(t) to get y(t)
y_t = convolve(x_t, h_t, mode='same') / len(t)

# Compute the mean of y(t)
mean_y_t = np.mean(y_t)
print("Mean of y(t):", mean_y_t)

# Compute the power spectral density (PSD) G_Y(f)
frequencies, GY_f = welch(y_t, nperseg=256, return_onesided=False)

# Shift the zero-frequency component to the center of the spectrum
frequencies = fftshift(frequencies)
GY_f = fftshift(GY_f)

# Plot the magnitude |GY(f)| vs. f
plt.figure(figsize=(10, 6))
plt.plot(frequencies, np.abs(GY_f))
plt.title('Magnitude of Power Spectral Density |GY(f)| vs. f')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude of PSD |GY(f)|')
plt.grid(True)
plt.show()
