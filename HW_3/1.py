import numpy as np
import matplotlib.pyplot as plot
import math

Nbits = 50000
Nsamp = 10
np.random.seed(30)
a = np.random.randint(0, 2, Nbits)
b = 2 * a - 1
Eb = 10

SNRdB_log = np.array([1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6])
sigma_log = [(1 / 2) * (10) * (10 ** (-snr / 10)) for snr in SNRdB_log]

def calculate_ber(mode, sigma):
    # Generate NRZ-L or Manchester Modulated signals
    x_t = []
    if mode == 'N':
        for i in range(Nbits):
            if a[i] == 1:
                x_t.extend([1] * Nsamp)
            else:
                x_t.extend([-1] * Nsamp)
    elif mode == 'M':
        for i in range(Nbits):
            if a[i] == 1:
                x_t.extend([1] * (Nsamp // 2) + [-1] * (Nsamp // 2))
            else:
                x_t.extend([-1] * (Nsamp // 2) + [1] * (Nsamp // 2))
    else:
        raise ValueError('Invalid mode')

    # Generate AWGN
    mu = 0
    n_t = np.random.normal(mu, sigma, Nbits * Nsamp)

    # Received signals
    r_t = np.array(x_t) + n_t

    # Correlator
    s_NRZL = np.array([1] * Nsamp)  # for NRZ-L
    s_Manchester = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, 1])  # for Manchester
    z = []
    if mode == 'N':
        for i in range(Nbits):
            z_t = np.multiply(r_t[i * Nsamp:(i + 1) * Nsamp], s_NRZL)
            z_t_out = sum(z_t)
            z.append(z_t_out)
    elif mode == 'M':
        for i in range(Nbits):
            z_t = np.multiply(r_t[i * Nsamp:(i + 1) * Nsamp], s_Manchester)
            z_t_out = sum(z_t)
            z.append(z_t_out)
    else:
        raise ValueError('Invalid mode')

    # Make decision, compare z with 0
    a_hat = [1 if zdata > 0 else 0 for zdata in z]

    # Calculate error
    err_num = sum(a != a_hat)

    # Calculate BER
    BER = err_num / Nbits
    return BER

# Calculate BER for each SNR value
BER_NRZL = [calculate_ber('N', sigma) for sigma in sigma_log]
BER_Manchester = [calculate_ber('M', sigma) for sigma in sigma_log]

# Plot the results
plot.figure(figsize=(10, 6))
plot.semilogy(SNRdB_log, BER_NRZL, marker='o', label='NRZ-L')
plot.semilogy(SNRdB_log, BER_Manchester, marker='x', label='Manchester')
plot.xlabel('SNR (dB)')
plot.ylabel('Bit Error Rate (BER)')
plot.legend()
plot.grid(True, which='both')
plot.title('BER vs. SNR for NRZ-L and Manchester Code')
plot.show()