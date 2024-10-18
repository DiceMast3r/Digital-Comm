import numpy as np
import matplotlib.pyplot as plt
import komm
import scipy.fft as fft


def SymbolToWave(symb, fc, t_symbol):
    # Generate the QPSK signal for each symbol and concatenate
    psk_signal = np.array([])  # Empty array to store the entire signal

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


def main_16SC(data, snr_in):
    M = 16  # QPSK modulation
    Nbit = len(data)
    f_1 = 5000  # 1st Carrier frequency (Hz)
    fs = f_1 * 10  # Sampling frequency (Hz)
    T = 2e-3  # Symbol duration (seconds)
    R_s = 1 / T  # Symbol rate (symbols/second)
    num_samples = int(fs * T)  # Number of samples per symbol
    t_symbol = np.linspace(
        0, T, num_samples, endpoint=False
    )  # Time vector for one symbol
    f_sc = ComputeSCFreq(f_1, M, R_s)

    data = np.array(data)
    while len(data) % 64 != 0:
        # add zero to the front of the data
        data = np.insert(data, 0, 0)
        Nbit = len(data)

    # QPSK modulation
    psk = komm.PSKModulation(M)
    qpsk_symb = psk.modulate(data)

    # Serial to 4 parallel output
    s_to_p_out = np.reshape(qpsk_symb, (16, Nbit // 64))

    ifft_data = np.array(fft.ifft2(s_to_p_out))  # IFFT of s_to_p_out

    # Parallel to serial output
    ifft_out = np.array(ifft_data).flatten()

    # Create a AWGN channel
    awgn = komm.AWGNChannel(snr=snr_in, signal_power="measured")
    rx_signal = awgn(ifft_out)
    np.round(rx_signal, 6)  # Add AWGN noise to the data

    # Serial to 4 parallel output
    # print("RX = ", rx_signal.round(3))
    rx_s_to_p_out = np.reshape(rx_signal, (16, Nbit // 64))

    # FFT of rx_data_2
    rx_fft = np.array(fft.fft2(rx_s_to_p_out))

    # 4 channel parallel to serial
    rx_fft_p_to_s = np.array(rx_fft).flatten()

    # Demodulate the received signal
    rx_bit = psk.demodulate(rx_fft_p_to_s)

    return rx_bit
