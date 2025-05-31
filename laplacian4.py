import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('cameraman.jpeg', cv2.IMREAD_GRAYSCALE)

# Define the Laplacian kernel
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

# Apply the Laplacian kernel to the image
laplacian_filtered = cv2.filter2D(image, -1, laplacian_kernel)

# Enhance the image by adding the Laplacian filtered image to the original image
enhanced_image = cv2.add(image, laplacian_filtered)

# Plot the original image, Laplacian filtered image, and enhanced image
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(laplacian_filtered, cmap='gray')
plt.title('Laplacian Filtered Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(enhanced_image, cmap='gray')
plt.title('Enhanced Image')
plt.axis('off')

plt.show()
