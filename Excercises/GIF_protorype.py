import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from PIL import Image

# Create a figure and axis
fig, ax = plt.subplots()

# Set the title and axis labels
ax.set_title("Animated Sine Wave")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")

# Example data: A simple sine wave that moves over time
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
line, = ax.plot(x, y)

# Update function for the animation
def update(frame):
    line.set_ydata(np.sin(x + frame / 10.0))  # Update the sine wave
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=100, blit=True)

# Save the animation as a GIF
ani.save("sine_wave_with_title.gif", writer='pillow', fps=10)
