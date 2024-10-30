import numpy as np
import matplotlib.pyplot as plot
import math
import scipy.fft as fft

Nsymb = 1200
Nsamp = 20
M = 4

#1. Generate data bits
np.random.seed(86)
a_I = np.random.randint(0,2,Nsymb)
I = 2*a_I -1

np.random.seed(87)
a_Q = np.random.randint(0,2,Nsymb)
Q = 2*a_Q -1

print("a_I = ", a_I)
print("a_Q = ", a_Q)

f = 10
t = np.arange(0, 1, 1/(f*Nsamp))
cos_t = np.cos(2*np.pi*f*t)
sin_t = np.sin(2*np.pi*f*t)
plot.figure(figsize=(10, 6))
plot.plot(t, cos_t, label='cos')
plot.plot(t, -1*sin_t, label='-sin')
plot.title("Carrier Wave")
plot.legend()

# Modulate

I_t=[]
Q_t=[]
for i in range(Nsymb):
   I_t.extend(I[i]*cos_t)
   Q_t.extend(-1*Q[i]*sin_t)

print(np.size(I_t))
print(np.size(Q_t))
print("I = ", I)
print("Q = ", Q)

x_t = np.add(I_t, Q_t)

#  if(a[i]==1)
#    x_t.extend(cos_t)
#  else:
#    x_t.extend(-1*cos_t)

tt = np.arange(0, Nsymb, 1/(f*Nsamp))

fig, axes = plot.subplots(3,1, figsize = (12,8))
axes[0].plot(I_t, 'g')
axes[0].grid(True)
axes[0].set_title('I(t) (I signal)')
axes[1].plot(Q_t, 'b')
axes[1].grid(True)
axes[1].set_title('Q(t) (Q signal)')
axes[2].plot(tt, x_t, 'r')
axes[2].grid(True)
axes[2].set_title('x(t) (QPSK signal)')
fig.tight_layout()

x_t_fft = np.array(fft.fft(x_t))
f_x_t = fft.fftfreq(len(x_t), 1/(f*Nsamp))

# Show Magnitude spectrum of x_t_fft
plot.figure(figsize=(12, 6))
plot.plot(f_x_t, np.abs(x_t_fft))
plot.title('Spectrum of QPSK Signal')
plot.xlabel('Frequency (Hz)')
plot.ylabel('Magnitude')
plot.grid(True)
plot.show()


#  Generate Gaussian noise
mu = 0
sigma = 4
n_t = np.random.normal(mu, sigma, np.size(x_t) )
plot.figure(figsize=(10, 6))
plot.plot(n_t)
plot.title("Gaussian Noise")
print("variance of noise = {:.3f}".format(np.var(n_t)))

# received signal
r_t = x_t +n_t

plot.figure(figsize=(10, 6))
plot.plot(r_t)
plot.title("Received Signal")

# Correlator

z_I = [ ]
z_I_tt = [ ]
z_Q=[ ]
z_Q_tt = [ ]

for i in range(Nsymb):
  z_I_t = np.multiply(r_t[i*f*Nsamp:(i+1)*f*Nsamp], cos_t)
  z_I_t = z_I_t/(f*Nsamp*0.5)
  z_I_t_out = sum(z_I_t)
  z_I_tt.extend(z_I_t)
  z_I.append(z_I_t_out)

  z_Q_t = np.multiply(r_t[i*f*Nsamp:(i+1)*f*Nsamp], -1*sin_t)
  z_Q_t = z_Q_t/(f*Nsamp*0.5)
  z_Q_t_out = sum(z_Q_t)
  z_Q_tt.extend(z_Q_t)
  z_Q.append(z_Q_t_out)


fig, axes = plot.subplots(2,1, figsize = (12,6))
axes[0].plot(z_I_tt, 'g')
axes[0].grid(True)
axes[0].set_title('z_I(t) (I signal)')
axes[1].plot(z_Q_tt, 'b')
axes[1].grid(True)
axes[1].set_title('z_Q(t) (Q signal)')
fig.tight_layout()

fig, axes = plot.subplots(2,1, figsize = (12,6))
axes[0].stem(z_I)
axes[0].grid(True)
axes[0].set_title('z_I')
axes[1].stem(z_Q)
axes[1].grid(True)
axes[1].set_title('z_Q')
fig.tight_layout()

print("z_I = ", [float("{:.3f}".format(i)) for i in z_I])
print("z_Q = ", [float("{:.3f}".format(i)) for i in z_Q])

plot.figure(figsize=(10, 6))
plot.scatter(z_I, z_Q, color='b')
plot.scatter([1,1,-1,-1],[1,-1,1,-1], color='r')
plot.title("Vector plot (Sigma = {0})".format(sigma))


# Make decision
a_I_hat = [ ]
for zdata in z_I:
  if (zdata > 0):
    a_I_hat.append(1)
  else:
    a_I_hat.append(0)

a_Q_hat = [ ]
for zdata in z_Q:
  if (zdata > 0):
    a_Q_hat.append(1)
  else:
    a_Q_hat.append(0)

"""print("estimated data")
print("a_I_hat = ", a_I_hat)
print("a_Q_hat = ", a_Q_hat)"""

print("transmitted data")
print('a_I = ', a_I)
print('a_Q = ', a_Q)

# Calculate the bit error rate
err_num_1 = sum((a_I != a_I_hat))
print('err_num I = ', err_num_1)
err_num_2 = sum((a_Q != a_Q_hat))
print('err_num Q = ', err_num_2)

ber = (err_num_1 + err_num_2) / (2*Nsymb)
print('BER = ', ber)

#plot.show()