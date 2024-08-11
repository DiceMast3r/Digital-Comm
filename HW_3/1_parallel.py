import numpy as np
import matplotlib.pyplot as plot
import time
from concurrent.futures import ThreadPoolExecutor

# Start the timer
start_time = time.time()

Nbits = 150000
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
    s_URZ = np.array([1,1,1,1,1,1,1,0,0,0]) #for Unipolar NRZ    
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

def calculate_ber_for_mode(mode):
    return [calculate_ber(mode, sigma) for sigma in sigma_log]

# Use ThreadPoolExecutor to calculate BER in parallel
with ThreadPoolExecutor() as executor:
    future_nrzl = executor.submit(calculate_ber_for_mode, 'N')
    future_manchester = executor.submit(calculate_ber_for_mode, 'M')
    future_polarrz = executor.submit(calculate_ber_for_mode, 'P')
    future_unipolarnrz = executor.submit(calculate_ber_for_mode, 'U')

    a_BER_NRZL = future_nrzl.result()
    BER_Manchester = future_manchester.result()
    BER_PolarRZ = future_polarrz.result()
    BER_UnipolarNRZ = future_unipolarnrz.result()

# End the timer
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print("Program execution time for {0} bits (Parallel): {1:.3f} seconds".format(Nbits, elapsed_time))

# Plot the results
plot.figure(figsize=(10, 6))
plot.semilogy(SNRdB_log, a_BER_NRZL, marker='o', label='NRZ-L')
plot.semilogy(SNRdB_log, BER_Manchester, marker='x', label='Manchester')
plot.semilogy(SNRdB_log, BER_PolarRZ, marker='s', label='Polar RZ')
plot.semilogy(SNRdB_log, BER_UnipolarNRZ, marker='d', label='Unipolar NRZ')
plot.xlabel('SNR (dB)')
plot.ylabel('Bit Error Rate (BER)')
plot.legend()
plot.grid(True, which='both')
plot.title('BER vs. SNR for NRZ-L and Manchester Code')
plot.show()