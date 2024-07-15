import numpy as np
import matplotlib.pyplot as plt

# Define the time vector
t = np.linspace(-30, 30, 8000)  # Adjust the range and number of points as needed


signal = 1/2 * ((24 * np.sinc(np.pi * 4 * (t - 20)) + (24 * np.sinc(np.pi * 4 * (t + 20)))) * np.exp(-1j * 2 * np.pi * 2 * t))

# Plotting the magnitude spectrum with increased resolution
plt.figure(figsize=(10, 6))
plt.plot(t, signal)
plt.title('Spectrum of the Signal 6 * rect((t-2) / 4) * cos(40 * pi * t)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()