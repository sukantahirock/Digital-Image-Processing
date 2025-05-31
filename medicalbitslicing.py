import cv2
import numpy as np
import matplotlib.pyplot as plt

def bitplane_slice(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the bitplanes
    bitplanes = [((gray_image >> i) & 1) * 255 for i in range(8)]

    return bitplanes

def plot_bitplanes_with_original(image, bitplanes):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    # Plot original image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Plot bitplanes
    axes = axes.flatten()[1:]
    for i in range(7):  # Adjusted to iterate up to 7
        axes[i].imshow(bitplanes[i], cmap='gray')
        axes[i].set_title(f'Bitplane {i}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# Load the medical image
image = cv2.imread('medical.png')

# Perform bitplane slicing
bitplanes = bitplane_slice(image)

# Plot the original image and bitplanes
plot_bitplanes_with_original(image, bitplanes)
