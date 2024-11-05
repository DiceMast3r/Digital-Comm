import numpy as np
import matplotlib.pyplot as plt
import komm
import scipy.fft as fft
from Module_SC import main_BPSK


def SymbolToWave(symb, fc, t_symbol):
    # Generate the QPSK signal for each symbol and concatenate
    psk_signal = np.array([])  # Empty array to store the entire signal

    for symbol in symb:
        # Extract I and Q components
        I = np.real(symbol)  # noqa: E741
        Q = np.imag(symbol)

        # Generate the modulated carrier for the current symbol
        carrier_I = I * np.cos(2 * np.pi * fc * t_symbol)
        carrier_Q = Q * np.sin(2 * np.pi * fc * t_symbol)

        # QPSK modulated signal for the current symbol
        symbol_signal = carrier_I - carrier_Q

        # Append to the overall QPSK signal
        psk_signal = np.concatenate((psk_signal, symbol_signal))

    return psk_signal


def ComputeSpectrum(sig, fs):
    # Compute the FFT of the signal
    sig_fft = np.array(fft.fft(sig))
    freq = fft.fftfreq(len(sig), 1 / fs)
    sig_fft /= len(sig)  # Normalize the FFT
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
    plt.scatter(psk.constellation.real, psk.constellation.imag, color="red")

    # Iterate over each symbol in the constellation
    for i, symb in enumerate(psk.constellation):
        gray_code = int_to_gray(i)  # Get the Gray code for the index
        plt.text(
            symb.real - 0.05, symb.imag + 0.05, f"{gray_code:02b}"
        )  # Display Gray code as binary

    plt.title("QPSK Constellation")
    plt.grid(True)
    plt.show()


# Parameters
M = 2  # QPSK modulation
Nsym = 2 * (1000)  # * Change number in parentheses *
Nbit = Nsym * 1
f_1 = 5000  # 1st Carrier frequency (Hz)
fs = f_1 * 10  # Sampling frequency (Hz)
T = 2e-3  # Symbol duration (seconds)
R_s = 1 / T  # Symbol rate (symbols/second)
num_samples = int(fs * T)  # Number of samples per symbol
t_symbol = np.linspace(0, T, num_samples, endpoint=False)  # Time vector for one symbol
f_sc = ComputeSCFreq(f_1, M, R_s)

np.random.seed(6)
data = np.random.randint(0, 2, Nbit)
# check if data is a multiple of 8 if not, add zeros to make it a multiple of 8
while len(data) % 2 != 0:
    # add zero to the front of the data
    data = np.insert(data, 0, 0)
    Nbit = len(data)
print("Data = ", data)
print("Data shape = ", data.shape)

# split even and odd bits
data_I = data[::2]
data_Q = data[1::2]

rx_bit_I = main_BPSK(data_I, 5)
rx_bit_Q = main_BPSK(data_Q, 5)

# Combine the rx_bit_I and rx_bit_Q to form the received data
rx_bit = np.zeros(Nbit, dtype=int)
rx_bit[::2] = rx_bit_I
rx_bit[1::2] = rx_bit_Q


print("Total bits: {0}, Error bits: {1}".format(Nbit, np.sum(rx_bit != data)))
print(f"Bit Error Rate: {ComputeBER(data, rx_bit, Nbit):.4f}")
