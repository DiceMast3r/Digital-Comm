import numpy as np
import matplotlib.pyplot as plot
import komm
import scipy.fft as fft

Nbit = 64 
Nsymb = Nbit // 2
Nsamp = 20
R_s = 8 # Symbol rate
T_s = 1 / R_s # Time period of one bit
M = 4

np.random.seed(86)
data = np.random.randint(0, 2, Nbit)

#print("Data bit = ", data)

# QPSK modulation
psk = komm.PSKModulation(M, phase_offset=np.pi/4)
data_1 = psk.modulate(data)
data_1_real = np.real(data_1)
data_1_imag = np.imag(data_1)

print("data_1 real = ", data_1_real)
print("data_1 imag = ", data_1_imag)

a_I = data_1_real
a_Q = data_1_imag

I = a_I 
Q = a_Q 

f = 10
t = np.arange(0, 1, 1/(f*Nsamp))
cos_t = np.cos(2*np.pi*f*t)
sin_t = np.sin(2*np.pi*f*t)
'''plot.figure(figsize=(10, 6))
plot.plot(t, cos_t, label='cos')
plot.plot(t, -1*sin_t, label='-sin')
plot.title("Carrier Wave")
plot.legend()
plot.show()'''

# Modulate

I_t=[]
Q_t=[]
for i in range(Nsymb):
   I_t.extend(I[i]*cos_t)
   Q_t.extend(-1*Q[i]*sin_t)

print("I = ", I)
print("Q = ", Q)

x_t = np.add(I_t, Q_t)

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
plot.show()

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
