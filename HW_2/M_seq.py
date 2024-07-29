import numpy as np
import matplotlib.pyplot as plt

def m_sequence_generator(seed):
    # Define the feedback taps for the shift register
    feedback_taps = [3, 0]  # Adjusted for 4-bit seed

    # Ensure feedback taps are within the bounds of the shift register
    if max(feedback_taps) > len(seed):
        raise ValueError("Feedback taps exceed the length of the seed")

    # Initialize the shift register with the seed
    shift_register = seed

    # Generate the M-sequence
    m_sequence = [0]
    while True:
        # Get the output bit by XORing the feedback taps
        output_bit = shift_register[feedback_taps[0] - 1] ^ shift_register[feedback_taps[1] - 1]

        # Append the output bit to the M-sequence
        m_sequence.append(output_bit)

        # Shift the register to the right
        shift_register = [output_bit] + shift_register[:-1]

        # Check if the M-sequence has repeated
        if shift_register == seed:
            break

    return m_sequence

# Example usage
seed = [0, 1, 1, 1]  # seed = 6
m_seq = m_sequence_generator(seed)
print(m_seq)

R1 = np.correlate(m_seq, m_seq, mode='full')
R1_norm = R1 / max(R1)

lag = np.arange(-len(m_seq) + 1, len(m_seq))

plt.figure(figsize=(10, 2))
plt.stem(lag, R1_norm)
plt.title('Correlation function: ID = {0}, Name = {1}'.format("65010386", "Thongchai Phanphaisan"))

plt.show()