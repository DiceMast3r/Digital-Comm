from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO

def image_to_binary_bits(image_url):
    # Download the image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert('L')  # Convert to grayscale
    # Convert image to numpy array
    image_array = np.array(image)
    # Flatten the array and convert to binary bits
    binary_bits = np.unpackbits(image_array.flatten())
    return binary_bits, image.size

def add_parity_bits(binary_bits, A):
    frames = []
    num_bits_per_frame = A * 8  # A bytes -> A * 8 bits
    for i in range(0, len(binary_bits), num_bits_per_frame):
        frame = binary_bits[i:i + num_bits_per_frame]
        parity_bit = 1 if np.sum(frame) % 2 != 0 else 0
        frames.append(np.append(frame, parity_bit))
    return np.concatenate(frames)

def polar_nrz_modulation(bits):
    return np.where(bits == 0, -1, 1)

def create_gaussian_noise(length, mean, var):
    return np.random.normal(mean, np.sqrt(var), length)

def received_signal(modulated_signal, noise):
    return modulated_signal + noise

def plot_histogram(received_signal):
    plt.hist(received_signal, bins=500, density=True, alpha=0.6, color='g')
    plt.xlabel('Amplitude')
    plt.ylabel('Probability Density')
    plt.title('Histogram of Received Signal')
    plt.grid(True)
    plt.show()

def reconstruct_image(received_signal, image_size, A):
    # Threshold the received signal
    thresholded_signal = np.where(received_signal > 0, 1, 0)
    # Remove parity bits
    num_bits_per_frame = A * 8
    reconstructed_bits = []
    for i in range(0, len(thresholded_signal), num_bits_per_frame + 1):
        frame = thresholded_signal[i:i + num_bits_per_frame]
        reconstructed_bits.extend(frame)
    reconstructed_bits = np.array(reconstructed_bits)
    # Convert binary bits back to pixel values
    pixel_values = np.packbits(reconstructed_bits)
    # Reshape to original image dimensions correctly
    reconstructed_image = pixel_values[:image_size[0] * image_size[1]].reshape(image_size[::-1])
    return reconstructed_image

    
def showImg(image_url):
    plt.imshow(np.array(Image.open(BytesIO(requests.get(image_url).content))))
    plt.title('Image')
    plt.show()

# Parameters
image_url = 'https://static.wikia.nocookie.net/witchers/images/4/4c/Haerin_OMG_Concept_Photo_%283%29.jpg/revision/latest/scale-to-width-down/512?cb=20230102104157'
A = 6  # Number of bytes per frame
variance = 0.3 + (A / 100)

# Steps
binary_bits, image_size = image_to_binary_bits(image_url)
framed_bits = add_parity_bits(binary_bits, A)
modulated_signal = polar_nrz_modulation(framed_bits)
noise = create_gaussian_noise(len(modulated_signal), 0, variance)
received_signal_1 = received_signal(modulated_signal, noise)
recon_img = reconstruct_image(received_signal_1, image_size, A)

plt.figure(figsize=(12, 5))
plt.hist(received_signal_1, bins=350, density=True, alpha=0.6, color='g')
plt.xlabel('Amplitude')
plt.ylabel('Probability Density')
plt.title('Histogram of Received Signal' + " with variance = {0:.5f}".format(variance))
plt.grid(True)

plt.figure(figsize=(5, 5))
plt.imshow(np.array(Image.open(BytesIO(requests.get(image_url).content))))
plt.title('Image (Original)')

plt.figure(figsize=(5, 5))
plt.imshow(recon_img)
plt.title('Image (Reconstructed with variance = {0:.5f})'.format(variance))

plt.show()