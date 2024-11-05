import numpy as np
import matplotlib.pyplot as plot
from scipy.special import erfc

Nsymb = 500
Nsamp = 20
M = 16
L = np.log2(M)

# Define inner and outer ring radii
r1 = 1
r2 = 2.85
k = []
phase_in = np.pi / 2  # Phase for inner ring (90°)
phase_out = np.pi / 6  # Phase for outer ring (30°)

# Generate constellation points for 16-APSK
ring1 = []
for i in range(4):
    ring1.append(r1 * np.exp(1j * (phase_in + 2 * np.pi * i / 4)))

ring2 = []
for i in range(12):
    ring2.append(r2 * np.exp(1j * (phase_out + 2 * np.pi * i / 12)))

apsk_con = np.array(ring1 + ring2)

# Randomly generate symbols
np.random.seed(35)
symbols = np.random.randint(0, M, Nsymb)

# Map symbols to constellation points
mapped_symbols = apsk_con[symbols]

# Generate I and Q components
I = np.real(mapped_symbols)
Q = np.imag(mapped_symbols)

f = 4
t = np.arange(0, 1, 1/(f*Nsamp))

cos_t = np.cos(2*np.pi*f*t)
sin_t = np.sin(2*np.pi*f*t)
plot.plot(t, cos_t)
plot.plot(t, sin_t)
plot.title("carrier signal")

# Modulate
I_t = []
Q_t = []

for i in range(Nsymb):
    I_t.extend(I[i] * cos_t)  # Modulate I signal
    Q_t.extend(Q[i] * sin_t)  # Modulate Q signal

I_t = np.array(I_t)
Q_t = np.array(Q_t)

# Combine I_t and Q_t
x_t = I_t + Q_t

# Create time vector tt matching the length of x_t
tt = np.arange(0, len(x_t) / (f * Nsamp), 1/(f * Nsamp))

# Plot signals
fig, axes = plot.subplots(3, 1, figsize=(12, 4))
axes[0].plot(I_t, 'g')
axes[0].grid(True)
axes[0].set_title('I(t) (I signal)')

axes[1].plot(Q_t, 'b')
axes[1].grid(True)
axes[1].set_title('Q(t) (Q signal)')

axes[2].plot(tt, x_t, 'r')
axes[2].grid(True)
axes[2].set_title('x(t) (APSK signal)')

fig.tight_layout()
plot.show()

# Add noise
mu = 0
sigma = 0.1
n_t = np.random.normal(mu, sigma, np.size(x_t))

r_t = x_t + n_t  # Adding noise to the transmitted signal
plot.plot(n_t)
plot.title('Noise signal')
plot.show()

plot.plot(r_t)
plot.title('Received signal')
plot.show()

# Correlator
z_I = []
z_Q = []
z_I_tt = []
z_Q_tt = []

for i in range(Nsymb):
    z_I_t = np.multiply(r_t[i*f*Nsamp:(i+1)*f*Nsamp], cos_t)
    z_I_t = z_I_t / (f*Nsamp*0.5)
    z_I_t_out = sum(z_I_t)
    z_I_tt.extend(z_I_t)
    z_I.append(z_I_t_out + np.random.normal(0, sigma))

    z_Q_t = np.multiply(r_t[i*f*Nsamp:(i+1)*f*Nsamp], -1*sin_t)
    z_Q_t = z_Q_t / (f*Nsamp*-1*0.5)
    z_Q_t_out = sum(z_Q_t)
    z_Q_tt.extend(z_Q_t)
    z_Q.append(z_Q_t_out + np.random.normal(0, sigma))

fig, axes = plot.subplots(3, 1, figsize=(12, 4))
axes[0].plot(r_t, 'g')
axes[0].set_title('APSK signal')
axes[0].grid(True)
axes[1].plot(z_I_tt, 'b')
axes[1].set_title('I signal')
axes[1].grid(True)
axes[2].plot(z_Q_tt, 'r')
axes[2].set_title('Q signal')
axes[2].grid(True)
plot.show()

# Plot constellation
plot.scatter(z_I, z_Q, color='b')
plot.scatter(np.real(apsk_con), np.imag(apsk_con), color='r')
plot.grid(True)
plot.title('16-APSK Constellation')
plot.show()

# Make decision
a_hat = []
for i in range(Nsymb):
    distances = np.abs(apsk_con - (z_I[i] + 1j*z_Q[i]))
    a_hat.append(np.argmin(distances))

print("Estimated data")
print(a_hat)

print("Transmitted data")
print(symbols)

snr_db = np.arange(0, 10, 1)  # SNR from 0 to 20 dB

# Calculate BER for APSK
def ber_apsk(M, snr_db):
    snr = 10 ** (snr_db / 10)
    ber = (1 / np.log2(M)) * erfc(np.sqrt(snr * (M - 1) / M))
    return ber

# Calculate BER
ber = ber_apsk(M, snr_db)

# Plot BER
plot.figure()
plot.semilogy(snr_db, ber, marker='o')
plot.title('BER for 16-APSK')
plot.xlabel('SNR (dB)')
plot.ylabel('BER')
plot.grid(True)
#plot.xlim(0, 20)
#plot.ylim(1e-5, 1)
plot.show()