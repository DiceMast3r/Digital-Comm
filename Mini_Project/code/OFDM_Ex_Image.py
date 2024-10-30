from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from Module_SC import main_4SC, main_8SC, main_16SC


def image_to_binary_bits(image_url, mode):
    # Download the image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert(mode) # Convert to grayscale or RGB
    # Convert image to numpy array
    image_array = np.array(image)
    # Flatten the array and convert to binary bits
    binary_bits = np.unpackbits(image_array.flatten())
    return binary_bits, image.size


def reconstruct_image(received_signal, image_size, mode):
    reconstructed_bits = []
    reconstructed_bits = np.array(received_signal)
    # Convert binary bits back to pixel values
    pixel_values = np.packbits(reconstructed_bits)
    # Reshape to original image dimensions correctly
    if mode == "L":
        reconstructed_image = pixel_values[: image_size[0] * image_size[1]].reshape(
            image_size[::-1]
        )
    elif mode == "RGB":
        reconstructed_image = pixel_values[: image_size[0] * image_size[1] * 3].reshape(
            image_size[::-1] + (3,)
        )
    return reconstructed_image


# Parameters
image_url = "https://static.wikia.nocookie.net/witchers/images/4/4c/Haerin_OMG_Concept_Photo_%283%29.jpg/revision/latest/scale-to-width-down/512?cb=20230102104157"
color_mode = "RGB"
snr = 8


# Steps
original_image = Image.open(BytesIO(requests.get(image_url).content))
binary_bits, image_size = image_to_binary_bits(image_url, mode=color_mode)
recon_img = reconstruct_image(binary_bits, image_size, mode=color_mode)

mod_bit_4SC = main_4SC(binary_bits, snr)
mod_bit_8SC = main_8SC(binary_bits, snr)
mod_bit_16SC = main_16SC(binary_bits, snr)
print(f"Bit length of modulated image: {len(mod_bit_4SC)}")
print(f"Bit shape of modulated image: {mod_bit_4SC.shape}")
recon_img_mod_4SC = reconstruct_image(mod_bit_4SC, image_size, mode=color_mode)
recon_img_mod_8SC = reconstruct_image(mod_bit_8SC, image_size, mode=color_mode)
recon_img_mod_16SC = reconstruct_image(mod_bit_16SC, image_size, mode=color_mode)

print(f"Bit length of image: {len(binary_bits)}")
print(f"Bit shape of image: {binary_bits.shape}")
print(f"Image size: {image_size}")

plt.figure(figsize=(5, 5))
plt.imshow(original_image)
plt.title("Original Image")

plt.figure()
plt.imshow(recon_img_mod_4SC)
plt.title("Reconstructed Image (4 Subcarriers)")
#plt.show()

plt.figure()
plt.imshow(recon_img_mod_16SC)
plt.title("Reconstructed Image (16 Subcarriers)")

plt.figure()
plt.imshow(recon_img_mod_8SC)
plt.title("Reconstructed Image (8 Subcarriers)")
plt.show()

