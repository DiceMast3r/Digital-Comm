import numpy as np
import matplotlib.pyplot as plt
import komm
import scipy.fft as fft
import time
import csv
import os
from Module_SC import main_BPSK


def BERCurve_QPSK(snr_in):
    def ComputeBER(data, rx_data, Nbit):
        # Compute the bit error rate
        ber = np.sum(rx_data != data) / Nbit
        return ber

    # Parameters
    M = 2  # PSK modulation
    Nsymb = 1000000  # Number of symbols
    Nbit = Nsymb * 1

    np.random.seed(6)
    data = np.random.randint(0, 2, Nbit)

    # split even and odd bits
    data_I = data[::2]
    data_Q = data[1::2]

    rx_bit_I = main_BPSK(data_I, snr_in)
    rx_bit_Q = main_BPSK(data_Q, snr_in)
    # Combine the rx_bit_I and rx_bit_Q to form the received data
    rx_bit = np.zeros(Nbit, dtype=int)
    rx_bit[::2] = rx_bit_I
    rx_bit[1::2] = rx_bit_Q
    return ComputeBER(data, rx_bit, Nbit)


def main():
    # Define the desired directory path
    save_directory = (
        "F:/Digital Comm/Mini_Project/csv"  # Change this to your desired directory
    )
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
    with open(file_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["SNR (dB)", "BER"])
        for snr_val, ber_val in zip(snr, ber_qpsk):
            csvwriter.writerow([snr_val, ber_val])


if __name__ == "__main__":
    main()
