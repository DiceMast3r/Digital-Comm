import numpy as np

import matplotlib.pyplot as plt

student_ID = int(input('Enter your student ID: '))
two_digit = student_ID % 100
freq = [int(x) for x in str(two_digit)]

# Generate time values from 0 to 5 seconds
t = np.linspace(0, 5, 500)

# Compute the cosine function
cos_a = np.cos(2 * np.pi * freq[0] * t)
cos_b = np.cos(2 * np.pi * freq[1] * t)
wave = cos_a + cos_b

# Plot the cosine function
print("last two digits of student ID:", two_digit)
plt.plot(t, wave, color='red')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Q27 plot for ' + str(student_ID))
plt.grid(True)
plt.show()