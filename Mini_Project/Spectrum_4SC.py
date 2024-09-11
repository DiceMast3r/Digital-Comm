import numpy as np
import matplotlib.pyplot as plt
import komm
import scipy.fft as fft

def SymbolToWave(symb, fc, t_symbol):
    # Generate the QPSK signal for each symbol and concatenate
    qpsk_signal = np.array([])  # Empty array to store the entire signal

    for symbol in symb:
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
        
    return qpsk_signal

def ComputeSpectrum(sig, fs):
    # Compute the FFT of the signal
    sig_fft = np.array(fft.fft(sig))
    freq = fft.fftfreq(len(sig), 1/fs)
    sig_fft /= len(sig) # Normalize the FFT
    return sig_fft, freq

def ComputeSCFreq(fc, N, R_s):
    # Compute the frequency of the subcarriers
    fc_n = []
    for i in range(N):
        fc_n.append(fc + (i * R_s))
    return fc_n

def ComputeBER(data, rx_data, Nbit):
    # Compute the bit error rate
    ber = np.sum(rx_data != data) / Nbit
    return ber

def int_to_gray(n):
    # Function to convert an integer to Gray code
    return n ^ (n >> 1)

def plot_constellation(psk):
    # Plot the constellation with Gray code labels
    plt.figure(figsize=(10, 5))
    plt.scatter(psk.constellation.real, psk.constellation.imag)
    
    # Iterate over each symbol in the constellation
    for i, symb in enumerate(psk.constellation):
        gray_code = int_to_gray(i)  # Get the Gray code for the index
        plt.text(symb.real - 0.05, symb.imag + 0.05, f"{gray_code:04b}")  # Display Gray code as binary
    
    plt.title('QPSK Constellation')
    plt.grid(True)
    plt.show()

# Parameters
M = 4  # QPSK modulation
Nbit = 1024 # Must be a multiple of 4
f_1 = 20 # 1st Carrier frequency (Hz)
fs = f_1 * 10  # Sampling frequency (Hz)
T = 0.25  # Symbol duration (seconds)
R_s = 1 / T  # Symbol rate (symbols/second)
num_samples = int(fs * T)  # Number of samples per symbol
t_symbol = np.linspace(0, T, num_samples, endpoint=False)  # Time vector for one symbol
f_sc = ComputeSCFreq(f_1, M, R_s)

np.random.seed(6)
data = np.random.randint(0, 2, Nbit)

# QPSK modulation
psk = komm.PSKModulation(M, phase_offset=np.pi/4)
qpsk_symb = psk.modulate(data)

# Serial to 4 parallel output
s_to_p_out = np.reshape(qpsk_symb, (4, Nbit // 8))
#print("Data 2 shape = ", s_to_p_out.shape)

ifft_data = np.array(fft.ifft2(s_to_p_out)) # IFFT of s_to_p_out

ifft_out = np.array(ifft_data).flatten()

print("Bit rate = {0} bits/second".format(R_s * np.log2(M)))
print("Symbol rate = {0} symbols/second".format(R_s))
print("Number of subcarriers: {0}".format(M))
print(f"Frequency of subcarrier: {ComputeSCFreq(f_1, M, R_s)} Hz")

# Extract the symbols to send to 4 subcarriers
sc = s_to_p_out[:4]

# Generate the QPSK signal for each symbol and concatenate
sig = []
for i in range(4):
    sig.append(SymbolToWave(sc[i], f_sc[i], t_symbol))

t_total = np.linspace(0, len(qpsk_symb) * T, len(sig) // 4, endpoint=False)

# Compute the spectrum of the modulated signal
sig_fft = []
freq = []
for i in range(4):
    sig_fft_i, freq_i = ComputeSpectrum(sig[i], fs)
    sig_fft.append(sig_fft_i)
    freq.append(freq_i)

# Plot the spectrum of the modulated signal
plt.figure(figsize=(10, 4))
for i in range(4):
    plt.plot(freq[i], np.abs(sig_fft[i]))
plt.title("Modulated Signal Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

# Create a AWGN channel
awgn = komm.AWGNChannel(snr=5, signal_power='measured') 
rx_signal = awgn(ifft_out); np.round(rx_signal, 6) # Add AWGN noise to the data

# Serial to 4 parallel output
rx_s_to_p_out = np.reshape(rx_signal, (4, Nbit // 8))

# FFT of rx_data_2
rx_fft = np.array(fft.fft2(rx_s_to_p_out)) 

# 4 channel parallel to serial
rx_fft_p_to_s = np.array(rx_fft).flatten()

# Demodulate the received signal
rx_bit = psk.demodulate(rx_fft_p_to_s)

print("Total bits: {0}, Error bits: {1}".format(Nbit, np.sum(rx_bit != data)))
print(f"Bit Error Rate: {ComputeBER(data, rx_bit, Nbit):.4f}")