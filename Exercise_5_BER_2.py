import numpy as np
import math

# Parameters
Nbits = 15
Nsamp = 10

# Generate the original signal
a = np.random.randint(0, 2, Nbits)
x_t = np.repeat(2 * a - 1, Nsamp)

for i in range(0, 20):
    sigma = i
    err_num = 0
    total_bits = 0

    while err_num < 100:
        # Generate AWGN
        mu = 0
        n_t = np.random.normal(mu, sigma, Nbits * Nsamp)

        # Received signals
        r_t = x_t + n_t

        # Correlator
        s_NRZL = np.array([1] * Nsamp)
        correlation = np.correlate(r_t, s_NRZL, mode='full')

        # A/D
        z = []
        for j in range(Nbits):
            zz = correlation[j * Nsamp + Nsamp - 1]
            z.append(zz)

        # Make decision, compare z with 0
        a_hat = []
        for zdata in z:
            if zdata > 0:
                a_hat.append(1)
            else:
                a_hat.append(0)

        # Compute error numbers
        err_num += sum(a != a_hat)
        total_bits += Nbits

    # Calculate BER
    BER = err_num / total_bits

    # Calculate Eb/N0
    Eb = np.mean(x_t**2)
    N0 = 2 * (sigma**2)
    EbN0 = 10 * math.log10(Eb / N0)

    print(f'SNR index: {i}, Errors: {err_num}, BER: {BER}, Eb/N0: {EbN0} dB')