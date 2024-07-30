import numpy as np
import matplotlib.pyplot as plt

# Define Eb/N0 values in dB
EbN0_dB = np.arange(1.6, 9.6, 0.5)

# Convert Eb/N0 from dB to linear scale
EbN0_linear = 10 ** (EbN0_dB / 10)

# Calculate noise variance and standard deviation for 4-PAM
# 4-PAM has 2 bits per symbol, so the energy per symbol is 2 * Eb
EsN0_linear = 2 * EbN0_linear
sigma2 = 1 / (2 * EsN0_linear)
sigma = np.sqrt(sigma2)

# Define number of symbols and errors threshold
num_symbols = int(1e6)
errors_threshold = 100

# Map bits to 4-PAM symbols: 00 -> -3, 01 -> -1, 11 -> 1, 10 -> 3
def bits_to_symbols(bits):
    symbols = -3 + 2 * bits[:, 0] + bits[:, 1]
    return symbols

# Map 4-PAM symbols back to bits
def symbols_to_bits(symbols):
    bits = np.zeros((symbols.size, 2), dtype=int)
    bits[symbols > 0, 0] = 1
    bits[symbols == 3, 1] = 1
    bits[symbols == -1, 1] = 1
    return bits

# Function to simulate 4-PAM transmission and calculate BER
def calculate_ber_4pam():
    ber = []
    for s in sigma:
        errors = 0
        total_bits = 0
        while errors < errors_threshold:
            # Generate random bits
            bits = np.random.randint(0, 2, (num_symbols, 2))
            
            # Map bits to 4-PAM symbols
            symbols = bits_to_symbols(bits)
            
            # Add Gaussian noise
            noise = np.random.normal(0, s, symbols.shape)
            received = symbols + noise
            
            # Demodulate and decode symbols to bits
            received_symbols = np.round(received / 2) * 2  # Correct rounding
            received_symbols = np.clip(received_symbols, -3, 3)
            received_bits = symbols_to_bits(received_symbols)
            
            # Calculate errors
            errors += np.sum(bits != received_bits)
            total_bits += 2 * num_symbols
        
        ber.append(errors / total_bits)
    
    return ber

# Calculate BER for 4-PAM
ber_4pam = calculate_ber_4pam()

# Plot the results
plt.figure()
plt.semilogy(EbN0_dB, ber_4pam, marker='o', label='4-PAM')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.legend()
plt.grid(True, which='both')
plt.title('BER vs. Eb/N0 for 4-PAM')
plt.show()