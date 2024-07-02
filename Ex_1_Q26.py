import numpy as np
import matplotlib.pyplot as plt

def generate_wave(bit, duration=0.25, freq=4, sample_rate=1000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.where(bit == 1, np.sin(2 * np.pi * freq * t), -np.sin(2 * np.pi * freq * t))

student_id = int(input('Enter your student ID: '))
bits = [int(x) for x in format(student_id % 100, '08b')] + [sum([int(x) for x in format(student_id % 100, '08b')]) % 2]

plt.figure(figsize=(12, 6))
for i, bit in enumerate(bits):
    plt.plot(np.linspace(i * 0.25, (i + 1) * 0.25, 250, endpoint=False), generate_wave(bit))

plt.title('Bit Sets Represented by Sine Waves')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()