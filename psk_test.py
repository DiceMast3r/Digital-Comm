import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

Nbits = 10
Nsamp = 20

# Ensure Nbits is even for 4PSK (2 bits per symbol)
if Nbits % 2 != 0:
    Nbits += 1

# Generate data bits
np.random.seed(10)
a = np.random.randint(0, 2, Nbits)

# Group bits into symbols (2 bits per symbol for 4PSK)
symbols = [a[i] * 2 + a[i + 1] for i in range(0, Nbits, 2)]

# Define 4PSK modulation phases: 00 -> 0, 01 -> π/2, 10 -> π, 11 -> 3π/2
phase_map = {
    0: 0,            # 00 -> 0 rad
    1: np.pi / 2,    # 01 -> π/2 rad
    2: np.pi,        # 10 -> π rad
    3: 3 * np.pi / 2 # 11 -> 3π/2 rad
}

# Reference constellation points
constellation_points = {
    (1, 0): [0, 0],     # 00
    (0, 1): [0, 1],     # 01
    (-1, 0): [1, 0],    # 10
    (0, -1): [1, 1],    # 11
}

f = 4
t = np.arange(0, 1, 1/(f*Nsamp))
carrier_cos = np.cos(2 * np.pi * f * t)
carrier_sin = np.sin(2 * np.pi * f * t)

# Modulate
x_t = []
for symbol in symbols:
    phase = phase_map[symbol]
    x_t.extend(np.cos(2 * np.pi * f * t + phase))

tt = np.arange(0, Nbits // 2, 1/(f*Nsamp))

# Generate Gaussian noise
mu = 0
sigma = 1

n_t = np.random.normal(mu, sigma, np.size(x_t))

# Received signal
r_t = x_t + n_t

# Correlator (extracting in-phase and quadrature components)
z_I = []
z_Q = []
for i in range(len(symbols)):
    z_t_I = np.multiply(r_t[i*f*Nsamp:(i+1)*f*Nsamp], carrier_cos)
    z_t_Q = np.multiply(r_t[i*f*Nsamp:(i+1)*f*Nsamp], carrier_sin)
    
    z_t_I = z_t_I / (f*Nsamp*0.5)  # In-phase component
    z_t_Q = z_t_Q / (f*Nsamp*0.5)  # Quadrature component
    
    z_I.append(sum(z_t_I))  # I-component at symbol period
    z_Q.append(sum(z_t_Q))  # Q-component at symbol period

plt.figure(figsize=(10, 5))
plt.stem(z_I, 'b', label='I')
plt.stem(z_Q, 'r', label='Q')
plt.legend()
plt.title('Correlator Output (I and Q Components)')

# Decision process
a_hat = []
for i in range(len(z_I)):
    # Find the nearest constellation point
    decision = min(constellation_points.keys(), 
                   key=lambda point: (z_I[i] - point[0])**2 + (z_Q[i] - point[1])**2)
    
    # Map the decision to the corresponding bits
    a_hat.extend(constellation_points[decision])

# Calculate the bit error rate
err_num = sum((a[:Nbits] != a_hat))
print('err_num = ', err_num)

ber = err_num / Nbits
print('BER = ', ber)

"""# Plot constellation diagram with decision points
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel('In-phase Component (I)')
ax.set_ylabel('Quadrature Component (Q)')
ax.set_title('4PSK Constellation Diagram')

# Red reference points for 4PSK: (1, 0), (0, 1), (-1, 0), (0, -1)
ax.scatter([1, 0, -1, 0], [0, 1, 0, -1], color='r')

# Scatter plot for the constellation points
scatter, = ax.plot([], [], 'bo')

def update(frame):
    scatter.set_data(z_I[:frame], z_Q[:frame])
    return scatter,

# Animate and save as GIF
ani = FuncAnimation(fig, update, frames=len(z_I), blit=True, interval=10)

ani.save('4psk_constellation_decision.gif', writer=PillowWriter(fps=30))"""

plt.show()
