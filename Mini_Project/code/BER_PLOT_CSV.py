import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV files
qpsk = pd.read_csv("F:/Digital Comm/Mini_Project/csv/ber_values_qpsk.csv")
_8psk = pd.read_csv("F:/Digital Comm/Mini_Project/csv/ber_values_8psk.csv")
_16psk = pd.read_csv("F:/Digital Comm/Mini_Project/csv/ber_values_16psk.csv")

# Plot the BER values
plt.figure(figsize=(10, 6))
plt.semilogy(qpsk['SNR (dB)'], qpsk['BER'], label='4 Subcarriers (QPSK)')
plt.semilogy(_8psk['SNR (dB)'], _8psk['BER'], label='8 Subcarriers (8PSK)')
plt.semilogy(_16psk['SNR (dB)'], _16psk['BER'], label='16 Subcarriers (16PSK)')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('BER vs SNR for different number of subcarriers')
plt.legend()
plt.grid(True, which='both')
plt.show()