import numpy as np
import matplotlib.pyplot as plt

# OFDM parameters
N = 8  # Number of subcarriers
Ncp = 2  # Length of cyclic prefix
data = np.random.randint(0, 2, 2 * N)  # Random binary data (0s and 1s), 2 bits per subcarrier

# Plot the data
plt.figure(figsize=(10, 4))
plt.stem(data)
plt.title('Random Data')
plt.xlabel('Bit Index')
plt.ylabel('Value')
plt.grid(True)

# QPSK modulation
# Mapping: 00 -> 1 + 1j, 01 -> -1 + 1j, 11 -> -1 - 1j, 10 -> 1 - 1j
modulated_data = np.zeros(N, dtype=complex)
for i in range(N):
    bits = data[2*i:2*i+2]
    if (bits == [0, 0]).all():
        modulated_data[i] = 1 + 1j
    elif (bits == [0, 1]).all():
        modulated_data[i] = -1 + 1j
    elif (bits == [1, 1]).all():
        modulated_data[i] = -1 - 1j
    elif (bits == [1, 0]).all():
        modulated_data[i] = 1 - 1j

# plot the modulated data
plt.figure(figsize=(10, 4))
plt.plot(modulated_data.real, label='Real Part')
plt.plot(modulated_data.imag, label='Imaginary Part', linestyle='--')
plt.title('QPSK Modulated Data')
plt.xlabel('Subcarrier Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# IFFT to convert to time domain
ifft_data = np.fft.ifft(modulated_data, N)

# Plotting the IFFT data
plt.figure(figsize=(10, 4))
plt.plot(ifft_data.real, label='Real Part')
plt.plot(ifft_data.imag, label='Imaginary Part', linestyle='--')
plt.title('IFFT Data (QPSK Modulation)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)


# Adding Cyclic Prefix
cyclic_prefix = ifft_data[-Ncp:]  # Take the last Ncp samples
ofdm_signal = np.concatenate([cyclic_prefix, ifft_data])  # Prepend cyclic prefix

# Channel (No noise for simplicity)
received_signal = ofdm_signal

# Removing Cyclic Prefix
received_signal = received_signal[Ncp:]

# FFT to convert back to frequency domain
fft_data = np.fft.fft(received_signal, N)

# plot the spectrum of the received signal
plt.figure(figsize=(10, 4))
plt.plot(fft_data)
plt.title('Spectrum of Received Signal (QPSK Modulation)')
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')
plt.grid(True)

# Demodulation (QPSK)
received_data = np.zeros(2 * N, dtype=int)
for i in range(N):
    real_part = np.real(fft_data[i])
    imag_part = np.imag(fft_data[i])
    if real_part > 0 and imag_part > 0:
        received_data[2*i:2*i+2] = [0, 0]
    elif real_part < 0 and imag_part > 0:
        received_data[2*i:2*i+2] = [0, 1]
    elif real_part < 0 and imag_part < 0:
        received_data[2*i:2*i+2] = [1, 1]
    elif real_part > 0 and imag_part < 0:
        received_data[2*i:2*i+2] = [1, 0]

# Print results
print("Original Data:      ", data)
print("Modulated Signal:   ", modulated_data)
print("OFDM Signal:        ", ofdm_signal)
print("Received Signal:    ", received_signal)
print("Demodulated Data:   ", received_data)

# Plotting the OFDM signal
plt.figure(figsize=(10, 4))
plt.plot(ofdm_signal.real, label='Real Part')
plt.plot(ofdm_signal.imag, label='Imaginary Part', linestyle='--')
plt.title('OFDM Signal with Cyclic Prefix (QPSK Modulation)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plotting the spectrum of the OFDM signal
plt.figure(figsize=(10, 4))
plt.plot(np.abs(np.fft.fft(ofdm_signal, 1024)))
plt.title('Spectrum of OFDM Signal (QPSK Modulation)')
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')
plt.grid(True)

plt.show()
