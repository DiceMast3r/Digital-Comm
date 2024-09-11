import numpy as np
import matplotlib.pyplot as plot
import math
from matplotlib.animation import FuncAnimation, PillowWriter


Nbits = 1000
Nsamp = 20
M = 2

#1. Generate data bits
np.random.seed(86)
a = np.random.randint(0,2,Nbits)
b = 2 * a - 1
print(a)
print(b)

f = 4
t = np.arange(0, 1, 1/(f*Nsamp))
cos_t = np.cos(2*np.pi*f*t)
plot.figure(figsize=(10, 6))
plot.plot(t, cos_t)
plot.title("Carrier Wave")

# Modulate
x_t = []
for i in range(Nbits):
   x_t.extend(b[i]*cos_t)
#  if(a[i]==1)
#    x_t.extend(cos_t)
#  else:
#    x_t.extend(-1*cos_t)

tt = np.arange(0, Nbits, 1/(f*Nsamp))
plot.figure(figsize=(10, 6))
plot.plot(tt, x_t)
plot.title("Modulated Signal")

#  Generate Gaussian noise
mu = 0
sigma = 0.1
n_t = np.random.normal(mu, sigma, np.size(x_t) )
plot.figure(figsize=(10, 6))
plot.plot(n_t)
plot.title("Gaussian Noise")
print("variance of noise = {:.3f}".format(np.var(n_t)))

# received signal
r_t = x_t + n_t

plot.figure(figsize=(10, 6))
plot.plot(r_t)
plot.title("Received Signal")

# Correlator

z = [ ]
z_tt = [ ]
for i in range(Nbits):
  z_t = np.multiply(r_t[i*f*Nsamp:(i+1)*f*Nsamp], cos_t)
  z_t = z_t/(f*Nsamp*0.5)
  z_t_out = sum(z_t)
  z_tt.extend(z_t)
  z.append(z_t_out)
  
print("z = ", [float("{:.3f}".format(i)) for i in z])

plot.figure(figsize=(10, 6))
plot.plot(z_tt)
plot.title("Correlator Output")

plot.figure(figsize=(10, 6))
plot.stem(z)
plot.title("Correlator Output (Stem Plot)")

plot.figure(figsize=(10, 6))
plot.scatter(z, np.zeros(Nbits), color='b')
plot.scatter([1,-1],[0, 0], color='r')
plot.title("Vector plot (Sigma = {0})".format(sigma))

plot.figure(figsize=(10, 6))
plot.hist(z, density=True, bins=100)
plot.title("Histogram of Correlator Output (Sigma = {0})".format(sigma))

# Make decision
a_hat = [ ]
for zdata in z:
  if (zdata > 0):
    a_hat.append(1)
  else:
    a_hat.append(0)

print("a_hat = ", a_hat)

plot.figure(figsize=(10, 6))
plot.stem(a_hat)
plot.title("Decoded Signal")

# Create the constellation plot GIF
fig, ax = plot.subplots()  # Create a figure and axis
ax.set_xlim(-2, 2)  # Set the limits of the plot
ax.set_ylim(-2, 2)  # Set the limits of the plot
ax.set_xlabel('In-phase Component')  # Label
ax.set_ylabel('Quadrature Component')  # Label
ax.set_title('Constellation Diagram')  # Title
ax.grid(True)  # Show grid

# Red reference points at [1, -1]
ax.scatter([1, -1], [0, 0], color='r')  # Reference points

# Scatter plot for the in-phase and quadrature components
scatter, = ax.plot([], [], 'bo')  # Blue dots

def update(frame):  # Update function for the animation
    scatter.set_data(z[:frame], np.zeros(frame))  # Update the scatter plot
    return scatter,  # Return the updated plot

# Animate and save as animated GIF
ani = FuncAnimation(fig, update, frames=200, blit=True, interval=20)  # frames = length of GIF, interval = time between frames in milliseconds

ani.save('BPSK_constellation.gif', writer=PillowWriter(fps=10))  # Save as animated GIF, fps = frames per second, save in the same directory as the script

#plot.show()