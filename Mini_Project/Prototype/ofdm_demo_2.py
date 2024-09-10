import numpy as np
import matplotlib.pyplot as plt

# Generate random bits
num_bits = 1000
bits = np.random.randint(0, 2, num_bits)

t = np.linspace(0, 1, num_bits)  # Time vector

# Define subcarriers
subcarrier1 = np.cos(2 * np.pi * 10 * t)
subcarrier2 = np.cos(2 * np.pi * 20 * t)

# FFT of subcarriers
fft_subcarrier1 = np.fft.fft(subcarrier1)
fft_subcarrier2 = np.fft.fft(subcarrier2)
plt.figure(figsize=(12, 6))
plt.plot(np.abs(fft_subcarrier1), label='Subcarrier 1')
plt.plot(np.abs(fft_subcarrier2), label='Subcarrier 2')
plt.title('FFT of Subcarriers')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)

plt.show()

