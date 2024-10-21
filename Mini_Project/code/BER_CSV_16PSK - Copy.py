import numpy as np
import matplotlib.pyplot as plt
import komm
import scipy.fft as fft
import time
import csv
import os

def BERCurve_16PSK(snr_in):
    def ComputeBER(data, rx_data, Nbit):
        # Compute the bit error rate
        ber = np.sum(rx_data != data) / Nbit
        return ber

    # Parameters
    M = 16  # 16PSK modulation
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

def main():
    # Define the desired directory path
    save_directory = "F:/Digital Comm/Mini_Project/csv"  # Change this to your desired directory
    file_name = "ber_values_16psk.csv"
    file_path = os.path.join(save_directory, file_name)

    # Start the timer
    start_time = time.time()

    snr = np.arange(0, 10, 0.1)
    snr_linear = 10 ** (snr / 10)
    ber_qpsk = [BERCurve_16PSK(snr_in) for snr_in in snr_linear]

    # End the timer
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print("(16PSK) Program execution time: {0:.3f} seconds".format(elapsed_time))

    # Save BER values to a CSV file
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['SNR (dB)', 'BER'])
        for snr_val, ber_val in zip(snr, ber_qpsk):
            csvwriter.writerow([snr_val, ber_val])
            
if __name__ == "__main__":
    main()
