import numpy as np
import matplotlib.pyplot as plt

student_ID = int(input('Enter your student ID: '))
freq = [5 if x == '0' else int(x) for x in str(student_ID % 100)]

t = np.linspace(0, 5, 500)
wave = np.cos(2 * np.pi * freq[0] * t) + np.cos(2 * np.pi * freq[1] * t)

plt.plot(t, wave, 'r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title(f'Q27 plot for {student_ID}')
plt.grid(True)
plt.show()