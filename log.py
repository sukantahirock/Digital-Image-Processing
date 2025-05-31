import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Load an example image
image = plt.imread('lena.png')

# Convert the image to grayscale if it's RGB
if len(image.shape) == 3:
    image = np.mean(image, axis=2)

# Define the Sobel operator for gradient computation
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

# Apply the Sobel operator to compute the gradients
gradient_x = convolve2d(image, sobel_x, mode='same', boundary='symm')
gradient_y = convolve2d(image, sobel_y, mode='same', boundary='symm')

# Compute the gradient magnitude
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

# Define the Laplacian operator
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

# Apply the Laplacian operator to the gradient magnitude to compute LoG
log_image = convolve2d(gradient_magnitude, laplacian_kernel, mode='same', boundary='symm')

# Plot the original image, gradient magnitude, and LoG
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Gradient Magnitude
plt.subplot(2, 3, 2)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.axis('off')

# LoG
plt.subplot(2, 3, 3)
plt.imshow(log_image, cmap='gray')
plt.title('LoG')
plt.axis('off')

# Histograms
plt.subplot(2, 3, 4)
plt.hist(image.ravel(), bins=256, color='black', alpha=0.5, histtype='stepfilled', density=True)
plt.title('Original Image Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Normalized Frequency')



plt.subplot(2, 3, 5)
plt.hist(gradient_magnitude.ravel(), bins=256, color='blue', alpha=0.5, histtype='stepfilled', density=True)
plt.title('Gradient Magnitude Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Normalized Frequency')

plt.subplot(2, 3, 6)
plt.hist(log_image.ravel(), bins=256, color='red', alpha=0.5, histtype='stepfilled', density=True)
plt.title('LoG Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Normalized Frequency')

plt.tight_layout()
plt.show()
