import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

Nbits = 100000
Nsamp = 20

# Generate data bits
np.random.seed(10)
a = np.random.randint(0, 2, Nbits)

f = 4
t = np.arange(0, 1, 1/(f*Nsamp))
cos_t = np.cos(2*np.pi*f*t)

# Modulate
x_t = []
for i in range(Nbits):
    x_t.extend(a[i] * cos_t)

tt = np.arange(0, Nbits, 1/(f*Nsamp))

# Generate Gaussian noise
mu = 0
sigma = 2.5

n_t = np.random.normal(mu, sigma, np.size(x_t))

# Received signal
r_t = x_t + n_t

# Correlator
z = []
z_tt = []
for i in range(Nbits):
    z_t = np.multiply(r_t[i*f*Nsamp:(i+1)*f*Nsamp], cos_t)
    z_t = z_t / (f*Nsamp*0.5)  # r(t)xcos of each bit period
    z_t_out = sum(z_t)  # output of summation/correlator at each bit period
    z_tt.extend(z_t)  # r(t)xcos at all time
    z.append(z_t_out)  # output of correlator at all time


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

# Create the constellation plot GIF
fig, ax = plt.subplots() # Create a figure and axis
ax.set_xlim(-1, 2) # Set the limits of the plot
ax.set_ylim(-1, 1) # Set the limits of the plot
ax.set_xlabel('In-phase Component') # Label
ax.set_ylabel('Quadrature Component') # Label
ax.set_title('Constellation Diagram') # Title
ax.grid(True) # Show grid

# Red reference points at [1, 0] and [0, 0]
ax.scatter([1, 0], [0, 0], color='r') # Reference points

# Scatter plot for the in-phase component
scatter, = ax.plot([], [], 'bo') # Blue dots

def update(frame):  # Update function for the animation
    scatter.set_data(z[:frame], np.zeros(frame)) # Update the scatter plot
    return scatter, # Return the updated plot

# Animate and save as animated GIF
ani = FuncAnimation(fig, update, frames=100, blit=True, interval=10) # frames = length of GIF in this case 100 frames, interval = time between frames in milliseconds

ani.save('constellation_with_ref.gif', writer=PillowWriter(fps=10)) # Save as animated GIF, fps = frames per second, save in the same directory as the script

plt.show()
