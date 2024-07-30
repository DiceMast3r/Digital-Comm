import numpy as np
import matplotlib.pyplot as plt


# Define Eb/N0 values in dB
A = 6
EbN0_dB = np.arange(1.6, 10.6, 1)

# Convert Eb/N0 from dB to linear scale
EbN0_linear = 10 ** (EbN0_dB / 10)

# Calculate noise variance and standard deviation
sigma2 = 1 / (2 * EbN0_linear)
sigma = np.sqrt(sigma2)

# Define number of bits and errors threshold
num_bits = int(1e3)
errors_threshold = 100

symbols = np.array([-3, -1, 1, 3])
Nsymbols = num_bits // 2  # Half as many symbols as bits

# Function to simulate transmission and calculate BER
def calculate_ber(line_code):
    ber = []
    for s in sigma:
        errors = 0
        total_bits = 0
        while errors < errors_threshold:
            # Generate random bits
            bits = np.random.randint(0, 2, num_bits)
            # Apply line code
            if line_code == 'NRZ-L':
                signal = 2 * bits - 1
            elif line_code == 'Manchester':
                signal = np.zeros(2 * num_bits)
                signal[::2] = 2 * bits - 1
                signal[1::2] = -(2 * bits - 1)
            elif line_code == 'Unipolar RZ':
                signal = np.zeros(2 * num_bits)
                signal[::2] = bits
            elif line_code == 'Polar RZ':
                signal = np.zeros(2 * num_bits)
                signal[::2] = 2 * bits - 1
            elif line_code == '4PAM':
                b = np.zeros(Nsymbols, dtype=int)
                for i in range(0, num_bits, 2):
                    idx = bits[i] * 2 + bits[i+1]
                    b[i//2] = symbols[idx]
                signal = np.repeat(b, Nsymbols)
                
            
            # Add Gaussian noise
            noise = np.random.normal(0, s, signal.shape)
            received = signal + noise
            
            # Demodulate and count errors
            if line_code == 'NRZ-L':
                received_bits = (received > 0).astype(int)
            elif line_code == 'Manchester':
                received_bits = (received[::2] > 0).astype(int)
            elif line_code == 'Unipolar RZ':
                received_bits = (received[::2] > 0.5).astype(int)
            elif line_code == 'Polar RZ':
                received_bits = (received[::2] > 0).astype(int)
            elif line_code == '4PAM':
                received_bits = np.zeros(num_bits, dtype=int)
                for i in range(0, num_bits, 2):
                    idx = np.argmin(np.abs(received[i] - symbols))
                    received_bits[i] = idx // 2
                    received_bits[i+1] = idx % 2
                
                
            errors += np.sum(bits != received_bits)
            total_bits += num_bits
        
        ber.append(errors / total_bits)
    
    return ber

# Calculate BER for each line code
ber_nrz_l = calculate_ber('NRZ-L')
ber_manchester = calculate_ber('Manchester')
ber_unipolar_rz = calculate_ber('Unipolar RZ')
ber_polar_rz = calculate_ber('Polar RZ')
ber_4pam = calculate_ber('4PAM')

# Plot the results
plt.figure()
plt.semilogy(EbN0_dB, ber_nrz_l, marker='o', label='NRZ-L')
plt.semilogy(EbN0_dB, ber_manchester, marker='x', label='Manchester')
plt.semilogy(EbN0_dB, ber_unipolar_rz, marker='s', label='Unipolar RZ')
plt.semilogy(EbN0_dB, ber_polar_rz, marker='d', label='Polar RZ')
plt.semilogy(EbN0_dB, ber_4pam, marker='^', label='4-PAM')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.legend()
plt.grid(True, which='both')
plt.title('BER vs. Eb/N0 for different line codes')
plt.show()
