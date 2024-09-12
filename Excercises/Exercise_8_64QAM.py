import numpy as np
import matplotlib.pyplot as plot
from matplotlib.animation import FuncAnimation, PillowWriter

Nsymb = 1000
Nsamp = 20
M = 64
L = np.log2(M)
# 1. Generate data bits
np.random.seed(86)
a_I = np.random.randint(0, 8, Nsymb)
I = 2 * a_I - (L - 1)

np.random.seed(87)
a_Q = np.random.randint(0, 8, Nsymb)
Q = 2 * a_Q - (L - 1)

print("a_I = ", a_I)
print("I = ", I)
print("a_Q = ", a_Q)
print("Q = ", Q)

f = 4
t = np.arange(0, 1, 1 / (f * Nsamp))
cos_t = np.cos(2 * np.pi * f * t)
sin_t = np.sin(2 * np.pi * f * t)
plot.figure(figsize=(10, 5))
plot.plot(t, cos_t)
plot.plot(t, -1 * sin_t)
plot.title("Carrier signals")
plot.legend(["cos(t)", "-sin(t)"])
plot.grid()
plot.show()

# Modulate

I_t = []
Q_t = []
for i in range(Nsymb):
    I_t.extend(I[i] * cos_t)
    Q_t.extend(-1 * Q[i] * sin_t)

print("I = ", I)
print("Q = ", Q)

x_t = np.add(I_t, Q_t)

tt = np.arange(0, Nsymb, 1 / (f * Nsamp))

fig, axes = plot.subplots(3, 1, figsize=(12, 4))
axes[0].plot(I_t, 'g')
axes[0].grid(True)
axes[0].set_title('I(t) (I signal)')
axes[1].plot(Q_t, 'b')
axes[1].grid(True)
axes[1].set_title('Q(t) (Q signal)')
axes[2].plot(tt, x_t, 'r')
axes[2].grid(True)
axes[2].set_title('x(t) (QAM signal)')
fig.tight_layout()
plot.show()

# Generate Gaussian noise
mu = 0
sigma = 1
n_t = np.random.normal(mu, sigma, np.size(x_t))
plot.figure(figsize=(10, 6))
plot.plot(n_t)
plot.title("Gaussian Noise")
print("variance of noise = {:.3f}".format(np.var(n_t)))
plot.show()

# Received signal
r_t = x_t + n_t

# Correlator

z_I = []
z_I_tt = []
z_Q = []
z_Q_tt = []

for i in range(Nsymb):
    z_I_t = np.multiply(r_t[i * f * Nsamp:(i + 1) * f * Nsamp], cos_t)
    z_I_t = z_I_t / (f * Nsamp * 0.5)
    z_I_t_out = sum(z_I_t)
    z_I_tt.extend(z_I_t)
    z_I.append(z_I_t_out)

    z_Q_t = np.multiply(r_t[i * f * Nsamp:(i + 1) * f * Nsamp], -1 * sin_t)
    z_Q_t = z_Q_t / (f * Nsamp * 0.5)
    z_Q_t_out = sum(z_Q_t)
    z_Q_tt.extend(z_Q_t)
    z_Q.append(z_Q_t_out)

fig, axes = plot.subplots(2, 1, figsize=(12, 4))
axes[0].plot(z_I_tt, 'g')
axes[0].grid(True)
axes[0].set_title('z_I(t) (I signal)')
axes[1].plot(z_Q_tt, 'b')
axes[1].grid(True)
axes[1].set_title('z_Q(t) (Q signal)')
fig.tight_layout()
plot.show()

fig, axes = plot.subplots(2, 1, figsize=(12, 4))
axes[0].stem(z_I)
axes[0].grid(True)
axes[0].set_title('z_I')
axes[1].stem(z_Q)
axes[1].grid(True)
axes[1].set_title('z_Q')
fig.tight_layout()
plot.show()

print("z_I = ", [float("{:.3f}".format(i)) for i in z_I])
print("z_Q = ", [float("{:.3f}".format(i)) for i in z_Q])

plot.figure(figsize=(10, 6))
plot.scatter(z_I, z_Q, color='b')
plot.scatter(I, Q, color='r')
plot.title("Vector plot (Sigma = {0})".format(sigma))
plot.grid(True, which='both')
plot.show()

# Make decision
a_I_hat = []
for zdata in z_I:
    if zdata > 6:
        a_I_hat.append(7)
    elif zdata > 4:
        a_I_hat.append(5)
    elif zdata > 2:
        a_I_hat.append(3)
    elif zdata > 0:
        a_I_hat.append(1)
    elif zdata > -2:
        a_I_hat.append(-1)
    elif zdata > -4:
        a_I_hat.append(-3)
    elif zdata > -6:
        a_I_hat.append(-5)
    else:
        a_I_hat.append(-7)

a_Q_hat = []
for zdata in z_Q:
    if zdata > 6:
        a_Q_hat.append(7)
    elif zdata > 4:
        a_Q_hat.append(5)
    elif zdata > 2:
        a_Q_hat.append(3)
    elif zdata > 0:
        a_Q_hat.append(1)
    elif zdata > -2:
        a_Q_hat.append(-1)
    elif zdata > -4:
        a_Q_hat.append(-3)
    elif zdata > -6:
        a_Q_hat.append(-5)
    else:
        a_Q_hat.append(-7)

print("estimated data")
print("a_I_hat = ", a_I_hat)
print("a_Q_hat = ", a_Q_hat)

print("transmitted data")
print('a_I = ', a_I)
print('a_Q = ', a_Q)

# Calculate the bit error rate
err_num_1 = sum((a_I != a_I_hat))
print('err_num I = ', err_num_1)
err_num_2 = sum((a_Q != a_Q_hat))
print('err_num Q = ', err_num_2)

ber = (err_num_1 + err_num_2) / (2 * Nsymb)
print('BER = ', ber)

plot.show()