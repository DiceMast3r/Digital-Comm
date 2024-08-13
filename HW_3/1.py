import numpy as np
import matplotlib.pyplot as plot
import time

# Start the timer
start_time = time.time()

Nbits = 200000
Nsamp = 10
np.random.seed(30)
a = np.random.randint(0, 2, Nbits)
b = 2 * a - 1
Eb = 10

SNRdB_log = np.array([1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6])
sigma_log = [np.sqrt((1 / 2) * (10) * (10 ** (-snr / 10))) for snr in SNRdB_log]

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
    elif mode == 'U':
        for i in range(Nbits):
            if a[i] == 1:
                x_t.extend([1] * (Nsamp // 2) + [0] * (Nsamp // 2))
            else:
                x_t.extend([0] * (Nsamp // 2) + [0] * (Nsamp // 2))
    elif mode == 'P':
        for i in range(Nbits):
            if a[i] == 1:
                x_t.extend([1] * (Nsamp // 2) + [0] * (Nsamp // 2))
            else:
                x_t.extend([-1] * (Nsamp // 2) + [0] * (Nsamp // 2))
    else:
        raise ValueError('Invalid mode')

    # Generate AWGN
    mu = 0
    n_t = np.random.normal(mu, sigma, Nbits * Nsamp)

    # Received signals
    r_t = np.array(x_t) + n_t

    # Correlator
    s_NRZL = np.array([1] * Nsamp)  # for NRZ-L
    s_Manchester = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])  # for Manchester
    s_PRZ = np.array([1,0,1,0,1,0,1,0,1,0]) #for Polar RZ
    s_URZ = np.array([1,1,1,1,1,0,0,0,0,0]) #for Unipolar RZ    
    z = []
    
    if mode == 'N':
        for i in range(Nbits):
            z_t = np.multiply(r_t[i * Nsamp:(i+1) * Nsamp], s_NRZL)
            z_t_out = sum(z_t)
            z.append(z_t_out)
    elif mode == 'M':
        for i in range(Nbits):
            z_t = np.multiply(r_t[i * Nsamp:(i+1) * Nsamp], s_Manchester)
            z_t_out = sum(z_t)
            z.append(z_t_out)
    elif mode == 'U':
        for i in range(Nbits):
            z_t = np.multiply(r_t[i * Nsamp:(i+1) * Nsamp], s_URZ)
            z_t_out = sum(z_t)
            z.append(z_t_out)
    elif mode == 'P':
        for i in range(Nbits):
            z_t = np.multiply(r_t[i * Nsamp:(i+1) * Nsamp], s_PRZ)
            z_t_out = sum(z_t)
            z.append(z_t_out)
    else:
        raise ValueError('Invalid mode')

    # Make decision, compare z with 0
    a_hat = [1 if zdata > 0 else 0 for zdata in z]

    # Calculate error
    err_num = sum(a != a_hat)

    # Calculate BER
    ber = err_num / Nbits
    return ber

# Calculate BER for each SNR value
a_BER_NRZL = [calculate_ber('N', sigma) for sigma in sigma_log]
BER_Manchester = [calculate_ber('M', sigma) for sigma in sigma_log]
BER_PolarRZ = [calculate_ber('P', sigma) for sigma in sigma_log]
BER_UnipolarNRZ = [calculate_ber('U', sigma) for sigma in sigma_log]

# End the timer
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print("Program execution time for {0} bits: {1:.3f} seconds".format(Nbits, elapsed_time))

# Plot the results
plot.figure(figsize=(10, 6))
plot.semilogy(SNRdB_log, a_BER_NRZL, marker='o', label='NRZ-L')
plot.semilogy(SNRdB_log, BER_Manchester, marker='x', label='Manchester')
plot.xlabel('SNR (dB)')
plot.ylabel('Bit Error Rate (BER)')
plot.legend()
plot.grid(True, which='both')
plot.title('BER vs. SNR for NRZ-L and Manchester')

plot.figure(figsize=(10, 6))
plot.semilogy(SNRdB_log, a_BER_NRZL, marker='o', label='NRZ-L')
plot.semilogy(SNRdB_log, BER_PolarRZ, marker='s', label='Polar RZ')
plot.xlabel('SNR (dB)')
plot.ylabel('Bit Error Rate (BER)')
plot.legend()
plot.grid(True, which='both')
plot.title('BER vs. SNR for NRZ-L and Polar RZ')

plot.figure(figsize=(10, 6))
plot.semilogy(SNRdB_log, a_BER_NRZL, marker='o', label='NRZ-L')
plot.semilogy(SNRdB_log, BER_UnipolarNRZ, marker='d', label='Unipolar RZ')
plot.xlabel('SNR (dB)')
plot.ylabel('Bit Error Rate (BER)')
plot.legend()
plot.grid(True, which='both')
plot.title('BER vs. SNR for NRZ-L and Unipolar RZ')

plot.figure(figsize=(10, 6))
plot.semilogy(SNRdB_log, a_BER_NRZL, marker='o', label='NRZ-L')
plot.semilogy(SNRdB_log, BER_Manchester, marker='x', label='Manchester')
plot.semilogy(SNRdB_log, BER_UnipolarNRZ, marker='d', label='Unipolar RZ')
plot.semilogy(SNRdB_log, BER_PolarRZ, marker='s', label='Polar RZ')
plot.xlabel('SNR (dB)')
plot.ylabel('Bit Error Rate (BER)')
plot.legend()
plot.grid(True, which='both')
plot.title('BER vs. SNR for Different Line Code')

plot.show()