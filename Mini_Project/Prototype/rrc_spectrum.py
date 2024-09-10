import numpy as np
import matplotlib.pyplot as plt

def rrc_filter(beta, span, sps):
    """
    Generates a Root Raised Cosine (RRC) filter.
    
    Parameters:
    beta (float): Roll-off factor (0 < beta <= 1)
    span (int): Filter span in symbols
    sps (int): Samples per symbol
    
    Returns:
    numpy.ndarray: RRC filter coefficients
    """
    n = span * sps  # total number of taps
    t = np.linspace(-span / 2, span / 2, n)  # time index for filter
    
    # Calculate the RRC filter response
    numerator = np.sin(np.pi * t * (1 - beta) / sps) + 4 * beta * t / sps * np.cos(np.pi * t * (1 + beta) / sps)
    denominator = np.pi * t * (1 - (4 * beta * t / sps) ** 2) / sps
    h = numerator / denominator
    
    # Handle singularity at t = 0 separately to avoid division by zero
    h[t == 0] = 1.0 - beta + (4 * beta / np.pi)
    
    # Handle singularities at t = ±sps/(4*beta) to avoid division by zero
    singularities = np.abs(t) == sps / (4 * beta)
    h[singularities] = (beta / np.sqrt(2)) * (
        ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))) + ((1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    )
    
    # Normalize the filter coefficients
    h /= np.sqrt(np.sum(h ** 2))
    
    return h, t

# Define RRC filter parameters
beta = 0.25   # Roll-off factor
span = 6      # Filter span in symbols
sps = 16      # Samples per symbol

# Generate RRC filter coefficients and time vector
rrc_coeffs, time = rrc_filter(beta, span, sps)

# Plotting the impulse response of the RRC filter
plt.figure(figsize=(12, 6))
plt.plot(time, rrc_coeffs, color='blue')
plt.title('Impulse Response of Root Raised Cosine Filter')
plt.xlabel('Time (t)')
plt.ylabel('h(t)')
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()
