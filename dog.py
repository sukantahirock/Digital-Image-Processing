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

# Apply Gaussian smoothing to the gradient magnitude
def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)

# Define the size and standard deviation of the Gaussian kernel
kernel_size = 5
sigma = 1.0

# Generate the Gaussian kernel
gaussian_kernel_2d = gaussian_kernel(kernel_size, sigma)

# Apply Gaussian smoothing to the gradient magnitude
smoothed_gradient_magnitude = convolve2d(gradient_magnitude, gaussian_kernel_2d, mode='same', boundary='symm')

# Compute the difference of gradients (DoG)
dog_image = gradient_magnitude - smoothed_gradient_magnitude

# Plot the original image, gradient magnitude, smoothed gradient magnitude, and DoG with histograms
plt.figure(figsize=(16, 12))

plt.subplot(3, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(3, 4, 5)
plt.hist(image.ravel(), bins=256, color='gray', alpha=0.7)
plt.title('Histogram')

plt.subplot(3, 4, 2)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.axis('off')
plt.subplot(3, 4, 6)
plt.hist(gradient_magnitude.ravel(), bins=256, color='gray', alpha=0.7)
plt.title('Histogram')

plt.subplot(3, 4, 3)
plt.imshow(smoothed_gradient_magnitude, cmap='gray')
plt.title('Smoothed Gradient Magnitude')
plt.axis('off')
plt.subplot(3, 4, 7)
plt.hist(smoothed_gradient_magnitude.ravel(), bins=256, color='gray', alpha=0.7)
plt.title('Histogram')

plt.subplot(3, 4, 4)
plt.imshow(dog_image, cmap='gray')
plt.title('Difference of Gradient (DoG)')
plt.axis('off')
plt.subplot(3, 4, 8)
plt.hist(dog_image.ravel(), bins=256, color='gray', alpha=0.7)
plt.title('Histogram')

plt.tight_layout()
plt.show()
