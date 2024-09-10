import numpy as np
import matplotlib.pyplot as plt
import komm
import scipy.fft as fft
import commpy as cp

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

# Function to convert an integer to Gray code
def int_to_gray(n):
    return n ^ (n >> 1)

def plot_constellation(psk):
    """ Plot the constellation with Gray code labels """
    plt.figure(figsize=(10, 5))
    plt.scatter(psk.constellation.real, psk.constellation.imag)
    
    # Iterate over each symbol in the constellation
    for i, symb in enumerate(psk.constellation):
        gray_code = int_to_gray(i)  # Get the Gray code for the index
        plt.text(symb.real - 0.05, symb.imag + 0.05, f"{gray_code:04b}")  # Display Gray code as binary
    
    plt.title('16-PSK Constellation')
    plt.grid(True)
    plt.show()

# Parameters
M = 16  # QPSK modulation
Nsymb = 16 * 100 # must be a multiple of 16
Nbit = Nsymb * 4 
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
psk = komm.PSKModulation(M, phase_offset=np.pi/16)
symb = psk.modulate(data)

# Serial to 16 parallel output
symb_s_to_p = np.reshape(symb, (16, Nbit // 64))

# IFFT of symb_s_to_p
ifft_data = np.array(fft.ifft2(symb_s_to_p))

# 16 Parallel to serial 
ifft_p_to_s_out = np.array(ifft_data).flatten()

print("Bit rate = {0} bits/second".format(R_s * np.log2(M)))
print("Symbol rate = {0} symbols/second".format(R_s))
#print(f"Frequency of subcarrier: {ComputeSCFreq(f_1, M, R_s)} Hz")

# Store symbols in a list of subcarriers
sc = symb_s_to_p[:16]

sig_1 = SymbolToWave(sc[0], f_sc[0], t_symbol)
sig_2 = SymbolToWave(sc[1], f_sc[1], t_symbol)
sig_3 = SymbolToWave(sc[2], f_sc[2], t_symbol)
sig_4 = SymbolToWave(sc[3], f_sc[3], t_symbol)
sig_5 = SymbolToWave(sc[4], f_sc[4], t_symbol)
sig_6 = SymbolToWave(sc[5], f_sc[5], t_symbol)
sig_7 = SymbolToWave(sc[6], f_sc[6], t_symbol)
sig_8 = SymbolToWave(sc[7], f_sc[7], t_symbol)
sig_9 = SymbolToWave(sc[8], f_sc[8], t_symbol)
sig_10 = SymbolToWave(sc[9], f_sc[9], t_symbol)
sig_11 = SymbolToWave(sc[10], f_sc[10], t_symbol)
sig_12 = SymbolToWave(sc[11], f_sc[11], t_symbol)
sig_13 = SymbolToWave(sc[12], f_sc[12], t_symbol)
sig_14 = SymbolToWave(sc[13], f_sc[13], t_symbol)
sig_15 = SymbolToWave(sc[14], f_sc[14], t_symbol)
sig_16 = SymbolToWave(sc[15], f_sc[15], t_symbol)

t_total = np.linspace(0, len(symb) * T, len(sig_1), endpoint=False)

sig_1_fft, freq_1 = ComputeSpectrum(sig_1, fs)
sig_2_fft, freq_2 = ComputeSpectrum(sig_2, fs)
sig_3_fft, freq_3 = ComputeSpectrum(sig_3, fs)
sig_4_fft, freq_4 = ComputeSpectrum(sig_4, fs)
sig_5_fft, freq_5 = ComputeSpectrum(sig_5, fs)
sig_6_fft, freq_6 = ComputeSpectrum(sig_6, fs)
sig_7_fft, freq_7 = ComputeSpectrum(sig_7, fs)
sig_8_fft, freq_8 = ComputeSpectrum(sig_8, fs)
sig_9_fft, freq_9 = ComputeSpectrum(sig_9, fs)
sig_10_fft, freq_10 = ComputeSpectrum(sig_10, fs)
sig_11_fft, freq_11 = ComputeSpectrum(sig_11, fs)
sig_12_fft, freq_12 = ComputeSpectrum(sig_12, fs)
sig_13_fft, freq_13 = ComputeSpectrum(sig_13, fs)
sig_14_fft, freq_14 = ComputeSpectrum(sig_14, fs)
sig_15_fft, freq_15 = ComputeSpectrum(sig_15, fs)
sig_16_fft, freq_16 = ComputeSpectrum(sig_16, fs)


plt.figure(figsize=(10, 4))
plt.plot(freq_1, np.abs(sig_1_fft))
plt.plot(freq_2, np.abs(sig_2_fft))
plt.plot(freq_3, np.abs(sig_3_fft))
plt.plot(freq_4, np.abs(sig_4_fft))
plt.plot(freq_5, np.abs(sig_5_fft))
plt.plot(freq_6, np.abs(sig_6_fft))
plt.plot(freq_7, np.abs(sig_7_fft))
plt.plot(freq_8, np.abs(sig_8_fft))
plt.plot(freq_9, np.abs(sig_9_fft))
plt.plot(freq_10, np.abs(sig_10_fft))
plt.plot(freq_11, np.abs(sig_11_fft))
plt.plot(freq_12, np.abs(sig_12_fft))
plt.plot(freq_13, np.abs(sig_13_fft))
plt.plot(freq_14, np.abs(sig_14_fft))
plt.plot(freq_15, np.abs(sig_15_fft))
plt.plot(freq_16, np.abs(sig_16_fft))
plt.title("Modulated Signal Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()


# Create a AWGN channel
awgn = komm.AWGNChannel(snr=10, signal_power='measured')
rx_signal = awgn(ifft_p_to_s_out); np.round(rx_signal, 6) # Add AWGN noise to the data

# Serial to 16 Parallel output
rx_s_to_p_out = np.reshape(rx_signal, (16, Nbit // 64))

# FFT of rx_data_2
rx_fft = np.array(fft.fft2(rx_s_to_p_out))

# 16 Parallel to serial
rx_fft_p_to_s_out = np.array(rx_fft).flatten()

# Demodulate the received signal
rx_bit = psk.demodulate(rx_fft_p_to_s_out)

print("Total bits: {0}, Error bits: {1}".format(Nbit, np.sum(rx_bit != data)))
print(f"Bit Error Rate: {ComputeBER(data, rx_bit, Nbit):.4f}")
