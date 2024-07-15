import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.fft import fft, fftfreq

def raised_cosine_filter(beta, sps, num_taps):
    """
    Generate a raised cosine filter (FIR) impulse response.

    :param beta: Roll-off factor (0 <= beta <= 1).
    :param sps: Samples per symbol.
    :param num_taps: Number of filter taps (must be odd).
    :return: Filter coefficients.
    """
    if beta == 0:
        # Special case for beta = 0 (sinc function)
        t = np.arange(-num_taps // 2, num_taps // 2 + 1) / sps
        h = np.sinc(t)
    else:
        t = np.arange(-num_taps // 2, num_taps // 2 + 1) / sps
        h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2)
        h[t == 0] = 1.0
        h[np.abs(t) == 1 / (2 * beta)] = np.pi / 4 * np.sinc(1 / (2 * beta))
    return h / np.sum(h)

def plot_spectrum(signal, sps, title):
    """
    Plot the spectrum of a signal.

    :param signal: Input signal.
    :param sps: Samples per symbol (used for x-axis scaling).
    :param title: Title for the plot.
    """
    N = len(signal)
    T = 1.0 / sps
    yf = fft(signal)
    xf = fftfreq(N, T)[:N // 2]

    plt.plot(np.array(xf), 2.0 / N * np.abs(np.array(yf[:N // 2])))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
def bits_to_4pam(bits):
    # Ensure the number of bits is even by adding a zero if necessary
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)
    
    # Convert bits to symbols
    symbols = []
    for i in range(0, len(bits), 2):
        two_bits = bits[i:i+2]
        symbol = (two_bits[0] * 2 + two_bits[1])  # Convert binary to decimal
        pam_symbol = 2 * symbol - 3  # Map to 4-PAM: -3, -1, 1, 3
        symbols.append(pam_symbol)
    
    return np.array(symbols)


# Parameters
beta = 0.5  # Roll-off factor
num_taps = 101  # Number of filter taps

# Generate random digital signal
num_symbols = 100
np.random.seed(42)

# Generate 10 random bits
data_bits = np.random.randint(0, 2, 10)
symbols = bits_to_4pam(data_bits)

# Assuming `symbols` has a length of 5 and you have a specific `sps` value
num_symbols = len(symbols)
sps = 20  # For example, 20 samples per symbol

# Correctly size the `upsampled_signal` array
upsampled_signal_length = num_symbols * sps
upsampled_signal = np.zeros(upsampled_signal_length)
# Now, this should work without raising an error
upsampled_signal[::sps] = symbols

# Generate raised cosine filter
rc_filter = raised_cosine_filter(beta, sps, num_taps)

# Apply raised cosine filter to the upsampled signal
shaped_signal = lfilter(rc_filter, 1, upsampled_signal)

# Plot original and shaped signals
plt.figure(figsize=(12, 10))

# Plot original upsampled signal
plt.subplot(3, 1, 1)
plt.step(np.arange(len(upsampled_signal)), upsampled_signal, where='post', label='Original Upsampled Signal', color='red')
plt.title('Original Upsampled Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Plot shaped signal
plt.subplot(3, 1, 2)
plt.plot(shaped_signal, label='Shaped Signal with Raised Cosine Filter')
plt.title('Shaped Signal with Raised Cosine Filter (Beta = {0})'.format(beta))
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Plot spectra
plt.subplot(3, 1, 3)
plot_spectrum(upsampled_signal, sps, 'Spectrum of Original Upsampled Signal')
plot_spectrum(shaped_signal, sps, 'Spectrum of Shaped Signal')
plt.legend(['Original Upsampled Signal', 'Shaped Signal'])

plt.tight_layout()
plt.show()
