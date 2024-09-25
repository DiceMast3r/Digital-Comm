import numpy as np
import matplotlib.pyplot as plt
import komm
import scipy.fft as fft
import time


def BERCurve_QPSK(snr_in):
    def ComputeBER(data, rx_data, Nbit):
        # Compute the bit error rate
        ber = np.sum(rx_data != data) / Nbit
        return ber

    # Parameters
    M = 4  # QPSK modulation
    Nsymb = 5000 # Number of symbols
    Nbit = Nsymb * 2 
    
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
    
    # Create a AWGN channel
    awgn = komm.AWGNChannel(snr=snr_in, signal_power='measured') 
    rx_signal = awgn(ifft_out); np.round(rx_signal, 6) # Add AWGN noise to the data

    # Serial to 4 parallel output
    rx_s_to_p_out = np.reshape(rx_signal, (4, Nbit // 8))

    # FFT of rx_data_2
    rx_fft = np.array(fft.fft2(rx_s_to_p_out)) 

    # 4 channel parallel to serial
    rx_fft_p_to_s = np.array(rx_fft).flatten()

    # Demodulate the received signal
    rx_bit = psk.demodulate(rx_fft_p_to_s)
    return ComputeBER(data, rx_bit, Nbit)


def BERCurve_16PSK(snr_in):
    def ComputeBER(data, rx_data, Nbit):
        # Compute the bit error rate
        ber = np.sum(rx_data != data) / Nbit
        return ber

    # Parameters
    M = 16  # QPSK modulation
    Nsymb = 16 * 5000 # must be a multiple of 16
    Nbit = Nsymb * 4 
    
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
    
    # Create a AWGN channel
    awgn = komm.AWGNChannel(snr=snr_in, signal_power='measured') 
    rx_signal = awgn(ifft_p_to_s_out); np.round(rx_signal, 6) # Add AWGN noise to the data

    # Serial to 16 Parallel output
    rx_s_to_p_out = np.reshape(rx_signal, (16, Nbit // 64))

    # FFT of rx_data_2
    rx_fft = np.array(fft.fft2(rx_s_to_p_out))

    # 16 Parallel to serial
    rx_fft_p_to_s_out = np.array(rx_fft).flatten()

    # Demodulate the received signal
    rx_bit = psk.demodulate(rx_fft_p_to_s_out)
    
    return ComputeBER(data, rx_bit, Nbit)

def BERCurve_8PSK(snr_in):
    def ComputeBER(data, rx_data, Nbit):
        # Compute the bit error rate
        ber = np.sum(rx_data != data) / Nbit
        return ber

    # Parameters
    M = 8  # QPSK modulation
    Nsymb = 8 * 5000 # must be a multiple of 8
    Nbit = Nsymb * 3 
    
    np.random.seed(6)
    data = np.random.randint(0, 2, Nbit)

    # QPSK modulation
    psk = komm.PSKModulation(M, phase_offset=np.pi/16)
    symb = psk.modulate(data)

    # Serial to 16 parallel output
    symb_s_to_p = np.reshape(symb, (8, Nbit // 24))

    # IFFT of symb_s_to_p
    ifft_data = np.array(fft.ifft2(symb_s_to_p))

    # 16 Parallel to serial 
    ifft_p_to_s_out = np.array(ifft_data).flatten()
    
    # Create a AWGN channel
    awgn = komm.AWGNChannel(snr=snr_in, signal_power='measured') 
    rx_signal = awgn(ifft_p_to_s_out); np.round(rx_signal, 6) # Add AWGN noise to the data

    # Serial to 16 Parallel output
    rx_s_to_p_out = np.reshape(rx_signal, (8, Nbit // 24))

    # FFT of rx_data_2
    rx_fft = np.array(fft.fft2(rx_s_to_p_out))

    # 16 Parallel to serial
    rx_fft_p_to_s_out = np.array(rx_fft).flatten()

    # Demodulate the received signal
    rx_bit = psk.demodulate(rx_fft_p_to_s_out)
    
    return ComputeBER(data, rx_bit, Nbit)
    


# Start the timer
start_time = time.time()

snr = np.arange(0.1, 10, 0.1)
snr_linear = 10 ** (snr / 10)
ber_qpsk = [BERCurve_QPSK(snr_in) for snr_in in snr_linear]
#ber_8psk = [BERCurve_8PSK(snr_in) for snr_in in snr_linear]
#ber_16psk = [BERCurve_16PSK(snr_in) for snr_in in snr_linear]

# End the timer
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print("Program execution time: {0:.3f} seconds".format(elapsed_time))

plt.figure(figsize=(10, 4))
plt.semilogy(snr, ber_qpsk, label='QPSK (4 subcarriers)')
#plt.semilogy(snr, ber_8psk, label='8-PSK')
#plt.semilogy(snr, ber_16psk, label='16-PSK')
plt.title("BER vs SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.grid(True, which='both')
plt.legend()
plt.show()