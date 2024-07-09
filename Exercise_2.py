import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fft_pack

Nbits = 100 + 6           # No. bits
Nsamp = 8             # No. samples/bits
 
#1. Generate data bits a, b #
np.random.seed(30 + 6)
a = np.random.randint(0 ,2 ,Nbits)
b = 2 * a - 1

# 2.Generate modulated signal m(t) #
m = []
for i in range(Nbits):
  if(a[i] == 1):
    m.extend([1]*Nsamp)
  else:
    m.extend([-1]*Nsamp)

m_fft = fft_pack.fft(m)
abs_m_fft = np.abs(m_fft)
f = np.linspace(0, Nsamp, Nbits * Nsamp)

print(len(a))
print(len(f))

#print("Data bits: ", a)
#print("Modulated signal: ", m)
# Assuming the previous code snippet has been executed

# 3. Create a time axis
t = np.arange(0, Nbits * Nsamp)

# 4. Plot the NRZ-L signal
plt.figure(figsize=(10, 2))

fig_1 = plt.figure(1)
plt.step(t, m, where='post')
plt.ylim(-1.5, 1.5)
plt.title('NRZ-L Line Coding')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.grid(True)

fig_2 = plt.figure(2)
plt.plot(f[0:400], (1/Nsamp) * abs_m_fft[0:400])
plt.title('NRZ-L')
plt.xlabel('f(Hz)')

plt.grid(True)
plt.show()



