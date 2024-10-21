import numpy as np
import matplotlib.pyplot as plt
import komm
import scipy.fft as fft
import time
import csv
import os

def BERCurve_QPSK(snr_in):
    def ComputeBER(data, rx_data, Nbit):
        # Compute the bit error rate
        ber = np.sum(rx_data != data) / Nbit
        return ber

    # Parameters
    M = 4  # QPSK modulation
    Nsymb = 100000 # Number of symbols
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
    

def main():
    # Define the desired directory path
    save_directory = "F:/Digital Comm/Mini_Project/csv"  # Change this to your desired directory
    file_name = "ber_values_qpsk.csv"
    file_path = os.path.join(save_directory, file_name)

    # Start the timer
    start_time = time.time()

    snr = np.arange(0, 10, 0.1)
    snr_linear = 10 ** (snr / 10)
    ber_qpsk = [BERCurve_QPSK(snr_in) for snr_in in snr_linear]

    # End the timer
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print("(QPSK) Program execution time: {0:.3f} seconds".format(elapsed_time))

    # Save BER values to a CSV file
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['SNR (dB)', 'BER'])
        for snr_val, ber_val in zip(snr, ber_qpsk):
            csvwriter.writerow([snr_val, ber_val])
            
if __name__ == '__main__':
    main()
