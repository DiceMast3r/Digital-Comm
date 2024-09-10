from math import log
import numpy as np
import matplotlib.pyplot as plt
import komm
import scipy.fft as fft

# Parameters
fc = 20  # Carrier frequency (Hz)
fs = fc * 20  # Sampling frequency (Hz)
T = 0.5 # Symbol duration (seconds)
num_samples = int(fs * T)  # Number of samples per symbol
t_symbol = np.linspace(0, T, num_samples, endpoint=False)  # Time vector for one symbol
M = 4
Nbit = 64
Nsymb = Nbit // log(M, 2)
Nsamp = 1
np.random.seed(86)
a = np.random.randint(0, 2, Nbit)

while len(a) % M != 0:
    # Append 0 at beginning of the array
    a = np.insert(a, 0, 0)

#print("Data bit = ", a)

psk = komm.PSKModulation(M, phase_offset=np.pi/4)

# Example QPSK symbol data (can replace with your actual data)
# Symbols are complex numbers (00 -> 1+0j, 01 -> 0+1j, 10 -> -1+0j, 11 -> 0-1j)
qpsk_symbol_data = psk.modulate(a)

# Generate the QPSK signal for each symbol and concatenate
qpsk_signal = np.array([])  # Empty array to store the entire signal

for symbol in qpsk_symbol_data:
    # Extract I and Q components
    I = np.real(symbol)
    Q = np.imag(symbol)
    
    # Generate the modulated carrier for the current symbol
    carrier_I = I * np.cos(2 * np.pi * fc * t_symbol)
    carrier_Q = Q * np.sin(2 * np.pi * fc * t_symbol)
    
    # QPSK modulated signal for the current symbol
    symbol_signal = carrier_I - carrier_Q
    
    # Append to the overall QPSK signal
    qpsk_signal = np.concatenate((qpsk_signal, symbol_signal))

# Time vector for the entire signal
t_total = np.linspace(0, len(qpsk_symbol_data) * T, len(qpsk_signal), endpoint=False)

qpsk_signal_fft = np.array(fft.fft(qpsk_signal))
freq = fft.fftfreq(len(qpsk_signal), 1/fs)
# Normalize the FFT
qpsk_signal_fft /= len(qpsk_signal)

# Plot the QPSK waveform
plt.figure(figsize=(10, 4))
plt.plot(t_total, qpsk_signal)
plt.title("QPSK Modulated Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Plot the FFT of the QPSK signal
plt.figure(figsize=(10, 4))
plt.plot(freq, np.abs(qpsk_signal_fft))
plt.title("FFT of QPSK Modulated Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()


awgn = komm.AWGNChannel(snr=6, signal_power='measured') # Add AWGN noise to the data
r_x = awgn(qpsk_symbol_data); np.round(r_x, 6)

rx_bit = psk.demodulate(r_x)
#print("Received data = ", rx_bit)

# Calculate the bit error rate
ber = np.sum(rx_bit != a) / Nbit
print(f"Bit Error Rate: {ber:.4f}")
