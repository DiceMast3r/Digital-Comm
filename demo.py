import komm 
import numpy as np
import matplotlib.pyplot as plt


Nsym = 100000
Nbit = Nsym * 2  # Number of bits

np.random.seed(42)  # Set seed for reproducibility
data = np.random.randint(0, 2, Nbit)  # Generate random data

print(f"Data length: {Nbit}")

# QPSK modulation

qpsk = komm.PSKModulation(4, phase_offset=np.pi / 4)  # Create a QPSK modulator
qpsk_symb = qpsk.modulate(data)  # Modulate the data


awgn = komm.AWGNChannel(snr=10, signal_power="1")
rx_data = awgn(qpsk_symb)

# QPSK demodulation
data_demod = qpsk.demodulate(rx_data)  # Demodulate the received data

ber = np.sum(data != data_demod) / Nbit  # Calculate the bit error rate
print(f"Bit error rate: {ber:.6f}")
