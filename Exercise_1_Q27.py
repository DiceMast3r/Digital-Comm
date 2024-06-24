import numpy as np

import matplotlib.pyplot as plt

# Generate time values from 0 to 5 seconds
t = np.linspace(0, 5, 500)

# Compute the cosine function
cosine = np.cos(2 * np.pi * t)

# Plot the cosine function
plt.plot(t, cosine)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Cosine Function')
plt.grid(True)
plt.show()