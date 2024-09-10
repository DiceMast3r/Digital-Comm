import numpy as np
import matplotlib.pyplot as plot
import komm
import scipy.fft as fft

Nbit = 1 * 16
np.random.seed(86)
data = np.random.randint(0, 2, Nbit)

#print("Data bit = ", data)

# QPSK modulation
psk = komm.PSKModulation(4, phase_offset=np.pi/4)
data_1 = psk.modulate(data)

'''print("data_1 size = ", data_1.size)
print("data_1 = ", data_1)'''

# Serial to 4 parallel output
data_2 = np.reshape(data_1, (4, Nbit // 8))
#print("Data 2 shape = ", data_2.shape)

ifft_data = np.array(fft.ifft2(data_2)) # IFFT of data_2
'''print("ifft_data = ", ifft_data)
print("ifft_data shape = ", ifft_data.shape)'''


'''# Show Magnitude spectrum of ifft_data
plot.figure(figsize=(12, 6))
plot.plot(np.abs(ifft_data))
plot.title('Magnitude Spectrum of IFFT Data')
plot.xlabel('Frequency (Hz)')
plot.ylabel('Magnitude')
plot.grid(True)
plot.show()'''


# 4 channel parallel to serial
data_3 = np.array(ifft_data).flatten()

# Ensure data_3 has the correct size before reshaping
if data_3.size != data_1.size:
    raise ValueError(f"Size of data_3 ({data_3.size}) does not match size of data_1 ({data_1.size})")

# Reshape data_3 to match the size of data_1
data_4 = np.reshape(data_3, data_1.shape)

print("data_4 = ", data_4)
print("data_4 shape = ", data_4.shape)

plot.figure(figsize=(12, 6))
plot.plot(data_4)
plot.title('Transmitted Signal')
plot.xlabel('Time [s]')
plot.ylabel('Amplitude')
plot.grid(True)
plot.show()

awgn = komm.AWGNChannel(snr=10, signal_power='measured') # Add AWGN noise to the data

rx_data = awgn(data_4); np.round(rx_data, 6)

plot.figure(figsize=(12, 6))
plot.plot(rx_data)
plot.title('Received Signal')
plot.xlabel('Time [s]')
plot.ylabel('Amplitude')
plot.grid(True)
plot.show()

#print("Received data = ", rx_data)

# Serial to 4 parallel output
rx_data_2 = np.reshape(rx_data, (4, Nbit // 8))
#print("Received Data 2 shape = ", rx_data_2.shape)

rx_fft = np.array(fft.fft2(rx_data_2)) # FFT of rx_data_2
#print("Received FFT = ", rx_fft)

'''# Show Magnitude spectrum of fft_data
plot.figure(figsize=(12, 6))
plot.plot(np.abs(rx_fft))
plot.title('Magnitude Spectrum of rx_fft Data')
plot.xlabel('Frequency (Hz)')
plot.ylabel('Magnitude')
plot.grid(True)
plot.show()'''


# 4 channel parallel to serial
rx_data_3 = np.array(rx_fft).flatten()
'''print("Received Data 3 shape = ", rx_data_3.shape)
print("Received Data 3 = ", rx_data_3)'''

plot.figure(figsize=(12, 6))
plot.plot(rx_data_3)
plot.title('Received Signal after FFT')
plot.xlabel('Time [s]')
plot.ylabel('Amplitude')
plot.grid(True)
plot.show()

rx_bit = psk.demodulate(rx_data_3)
#print("Received Bit = ", rx_bit)

plot.figure(figsize=(12, 6))
plot.plot(rx_bit)
plot.title('Received Bit')
plot.xlabel('Time [s]')
plot.ylabel('Amplitude')
plot.grid(True)
plot.show()

# Calculate the bit error rate
ber = np.sum(data != rx_bit) / Nbit
print(f"Bit Error Rate: {ber:.4f}")