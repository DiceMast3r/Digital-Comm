import numpy as np
import matplotlib.pyplot as plt

# Define the time vector
t = np.linspace(-10, 10, 1000)  # Adjust the range and number of points as needed

# Define the rect function
def rect(t):
    return np.where(np.abs(t) <= 0.5, 1, 0)

# Define the signal
signal = 6 * rect((t-2) / 4)

# Define the function to multiply with
function_to_multiply = np.cos(40 * np.pi * t)

# Perform the multiplication
result = signal * function_to_multiply

# Compute the FFT of the result
result_fft = np.fft.fft(result)

# Compute the frequency bins, assuming uniform sampling
sampling_rate = 1 / (t[1] - t[0])
freqs = np.fft.fftfreq(len(result), 1/sampling_rate)

# Apply a window function to the signal
windowed_signal = result * np.hamming(len(result))

# Increase the number of points in the FFT for higher resolution
n_fft = len(result) * 4  # Zero-padding for smoother spectrum

# Compute the FFT of the windowed signal with increased resolution
result_fft = np.fft.fft(windowed_signal, n=n_fft)

# Compute the frequency bins for the increased resolution
freqs = np.fft.fftfreq(n_fft, 1/sampling_rate)

# Plotting the magnitude spectrum with increased resolution
plt.figure(figsize=(10, 6))
plt.plot(freqs, np.abs(result_fft))
plt.title('Magnitude Spectrum of the Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()