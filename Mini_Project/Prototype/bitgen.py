import numpy as np
import matplotlib.pyplot as plot
import komm
import scipy.fft as fft

Nbit = 1024 
np.random.seed(86)
data = np.random.randint(0, 2, Nbit)

# QPSK modulation
psk = komm.PSKModulation(4, phase_offset=np.pi/4)
data_1 = psk.modulate(data)

#print("data_1 size = ", data_1.size)

# Serial to 4 parallel output
data_2 = np.reshape(data_1, (4, Nbit // 8))

ifft_data = fft.fft2(data_2)
#print("ifft_data = ", ifft_data)

# 4 channel parallel to serial
data_3 = np.array(ifft_data).flatten()

# Ensure data_3 has the correct size before reshaping
if data_3.size != data_1.size:
    raise ValueError(f"Size of data_3 ({data_3.size}) does not match size of data_1 ({data_1.size})")

# Reshape data_3 to match the size of data_1
data_4 = np.reshape(data_3, data_1.shape)

'''print("data_4 = ", data_4)
print("data_4 shape = ", data_4.shape)'''

# Assuming IFFT data is complex and already available as `ifft_data`
# Using a carrier frequency f_c and sampling frequency f_s
fc = 5e3  # Carrier frequency (in Hz)
fs = 10e3  # Sampling frequency (in Hz)
t = np.arange(len(np.array(ifft_data).flatten())) / fs  # Time vector based on sampling frequency

# Separate real and imaginary parts of the IFFT output
real_part = np.real(np.array(ifft_data))
imag_part = np.imag(np.array(ifft_data))

# Generate carrier signals
carrier_cos = np.cos(2 * np.pi * fc * t)  # In-phase carrier (cosine)
carrier_sin = np.sin(2 * np.pi * fc * t)  # Quadrature carrier (sine)

# Modulate the real and imaginary parts onto the carriers
modulated_real = real_part.flatten() * carrier_cos
modulated_imag = imag_part.flatten() * carrier_sin

# Combine the in-phase and quadrature components to get the final modulated signal
s_t = modulated_real - modulated_imag  # Transmitted signal

# Plot the real and imaginary parts of the modulated signal
plot.figure(figsize=(12, 6))
plot.subplot(2, 1, 1)
plot.plot(t, modulated_real, label='In-phase (Real part)')
plot.title('In-phase Component')
plot.xlabel('Time [s]')
plot.ylabel('Amplitude')
plot.grid(True)
plot.legend()

plot.subplot(2, 1, 2)
plot.plot(t, modulated_imag, label='Quadrature (Imaginary part)', color='r')
plot.title('Quadrature Component')
plot.xlabel('Time [s]')
plot.ylabel('Amplitude')
plot.grid(True)
plot.legend()

plot.tight_layout()
plot.show()

# Final transmitted signal s(t)
plot.figure(figsize=(10, 4))
plot.plot(t, s_t, label='Transmitted Signal s(t)')
plot.title('Transmitted Signal')
plot.xlabel('Time [s]')
plot.ylabel('Amplitude')
plot.grid(True)
plot.legend()
plot.show()

# Apply a window function (Hanning window) to reduce spectral leakage
window = np.hanning(len(s_t))
s_t_windowed = s_t * window

# Compute the FFT of the transmitted signal s_t using scipy
spectrum = fft.fft(s_t_windowed)
spectrum_magnitude = np.abs(np.array(spectrum))

# Generate frequency axis using scipy
freqs = fft.fftfreq(len(s_t), 1/fs)

# Plot the spectrum (magnitude vs. frequency)
plot.figure(figsize=(10, 5))
plot.plot(freqs[:len(freqs)//2], 20 * np.log10(spectrum_magnitude[:len(freqs)//2]))  # Convert to dB scale
plot.title('Spectrum of the Transmitted Signal with Windowing')
plot.xlabel('Frequency [Hz]')
plot.ylabel('Magnitude [dB]')
plot.grid(True)
plot.xlim(0, fs/2)  # Show only the positive frequency components
plot.show()