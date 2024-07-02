import numpy as np

import matplotlib.pyplot as plt

# convert a digit to its 4-bit binary representation
def digit_to_bits(digit):
    if digit == 0:
        return [0, 1, 0, 1]
    bits = [int(x) for x in format(digit, '04b')] # 04b means 4 bits with leading zeros
    return bits

# add even parity bit
def add_even_parity(bits):
    parity = sum(bits) % 2 # if bit is even, parity is 0; if bit is odd, parity is 1
    return bits + [parity]

# generate sine wave based on the bit value
def generate_wave(bit, duration, freq, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False) # generate time values
    if bit == 1:
        return np.sin(2 * np.pi * freq * t)
    else:
        return -np.sin(2 * np.pi * freq * t)

# Get the last two digits of the student ID
student_id = input('Enter your student ID: ')

# Assuming student_id is already defined
student_id_str = str(student_id)  # Convert student ID to string
last_two_digits_str = student_id_str[-2:]  # Get the last two characters

# Validate and convert
if last_two_digits_str.isdigit():
    last_two_digits = [int(x) for x in last_two_digits_str]
else:
    print("Error: Last two digits contain non-numeric characters.")
    last_two_digits = []

# Convert each digit to bits and add parity bit
digits = [int(x) for x in last_two_digits]
bit_sets = [add_even_parity(digit_to_bits(digit)) for digit in digits] # convert to bits and add parity bit in one line

# Parameters
duration = 0.25  # period is 0.25 seconds
freq = 4  # 4 Hz
sample_rate = 1000  # samples per second

# Plotting
plt.figure(figsize=(12, 6))
current_time = 0

for bit_set in bit_sets:
    for bit in bit_set:
        wave = generate_wave(bit, duration, freq, sample_rate)
        t = np.linspace(current_time, current_time + duration, int(sample_rate * duration), endpoint=False)
        plt.plot(t, wave)
        current_time += duration

plt.title('Bit Sets Represented by Sine Waves')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
