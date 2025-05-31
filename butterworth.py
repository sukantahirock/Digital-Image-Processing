import cv2
import numpy as np
from matplotlib import pyplot as plt

def butterworth_lowpass_filter(shape, cutoff, n):
    rows, cols = shape
    x = np.arange(cols)
    y = np.arange(rows)
    x -= cols // 2
    y -= rows // 2
    xv, yv = np.meshgrid(x, y)
    d = np.sqrt(xv**2 + yv**2)
    h = 1 / (1 + (d / cutoff)**(2*n))
    return h

def apply_butterworth_lowpass_filter(image, cutoff, n):
    fft_image = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_image)
    filtered_image = fft_shifted * butterworth_lowpass_filter(image.shape, cutoff, n)
    filtered_image = np.fft.ifftshift(filtered_image)
    filtered_image = np.fft.ifft2(filtered_image)
    filtered_image = np.abs(filtered_image)
    return filtered_image.astype(np.uint8)

# Load the image
image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

# Parameters for the Butterworth lowpass filter
cutoff_frequencies = [10, 30, 50]  # Different cutoff frequencies to test
orders = [1, 2, 4]  # Different orders to test

# Plotting setup
num_rows = len(cutoff_frequencies)
num_cols = len(orders)
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

for i, cutoff in enumerate(cutoff_frequencies):
    for j, order in enumerate(orders):
        # Apply the Butterworth lowpass filter
        filtered_image = apply_butterworth_lowpass_filter(image, cutoff, order)

        # Display the filtered image
        ax = axes[i, j]
        ax.imshow(filtered_image, cmap='gray')
        ax.set_title(f'Cutoff={cutoff}, Order={order}')
        ax.axis('off')

plt.tight_layout()
plt.show()