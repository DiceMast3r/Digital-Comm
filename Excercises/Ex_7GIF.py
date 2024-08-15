import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import math

Nbits = 10000
Nsamp = 20

# 1. Generate data bits
np.random.seed(10)
a = np.random.randint(0, 2, Nbits)
# print(a)

f = 4
t = np.arange(0, 1, 1/(f*Nsamp))
cos_t = np.cos(2*np.pi*f*t)
"""plt.figure(figsize=(10, 5))
plt.plot(t, cos_t)
plt.title("Cosine wave")"""

# Modulate
x_t = []
for i in range(Nbits):
    x_t.extend(a[i] * cos_t)

tt = np.arange(0, Nbits, 1/(f*Nsamp))
"""plt.figure(figsize=(10, 5))
plt.plot(tt, x_t)
plt.title("ASK Modulated signal x(t)")"""

# Generate Gaussian noise
mu = 0
sigma = 1

n_t = np.random.normal(mu, sigma, np.size(x_t))
print(np.var(n_t))
"""plt.figure(figsize=(10, 5))
plt.plot(n_t)
plt.title("Gaussian noise")"""

# Received signal
r_t = x_t + n_t

"""plt.figure(figsize=(10, 5))
plt.plot(r_t)
plt.title("Received signal r(t)")"""

# Correlator
z = []
z_tt = []
for i in range(Nbits):
    z_t = np.multiply(r_t[i*f*Nsamp:(i+1)*f*Nsamp], cos_t)
    z_t = z_t / (f*Nsamp*0.5)  # r(t)xcos of each bit period
    z_t_out = sum(z_t)  # output of summation/correlator at each bit period
    z_tt.extend(z_t)  # r(t)xcos at all time
    z.append(z_t_out)  # output of correlator at all time

plt.figure(figsize=(10, 5))
plt.plot(z_tt)
plt.title("Correlator output z(t)")

plt.figure(figsize=(10, 5))
plt.stem(z)
plt.title("Correlator output z[k]")

# Plot signal vectors, constellation of correlator output z
plt.figure(figsize=(10, 5))
plt.scatter(z, np.zeros(Nbits), color='b')
plt.scatter([1, 0], [0, 0], color='r')
plt.title("Constellation of correlator output z")

plt.figure(figsize=(10, 5))
plt.hist(z, density=True, bins=100)
plt.title("Histogram of correlator output z (Sigma = {0})".format(sigma))

# Make decision, compare z with threshold
a_hat = []
threshold = 0.5
for zz_1 in z:
    if zz_1 > threshold:
        a_hat.append(1)
    else:
        a_hat.append(0)

# Calculate the bit error rate
err_num = sum((a != a_hat))
print('err_num = ', err_num)

ber = err_num / Nbits
print('BER = ', ber)

# Create a figure and axis for the animation
fig, ax = plt.subplots(figsize=(10, 5))

def init():
    ax.clear()
    ax.set_title("Constellation of correlator output z")
    ax.scatter([1, 0], [0, 0], color='r')
    return ax,

def update(frame):
    ax.set_title("Constellation of correlator output z (Frame {0})".format(frame))
    ax.scatter(z[:frame], np.zeros(frame), color='b')
    ax.scatter([1, 0], [0, 0], color='r')
    return ax,

# Create the animation
ani = FuncAnimation(fig, update, frames=(Nbits // 10), init_func=init, blit=True)

# Save the animation as a GIF file using Pillow
ani.save('constellation_animation_1.gif', writer=PillowWriter(fps=30))

plt.show()