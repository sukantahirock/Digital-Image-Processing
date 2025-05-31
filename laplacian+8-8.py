import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('cameraman.jpeg', cv2.IMREAD_GRAYSCALE)

# Define the Laplacian kernel for +8 and -8
laplacian_kernel_pos8 = np.array([[1, 1, 1],
                                  [1, -8, 1],
                                  [1, 1, 1]])

laplacian_kernel_neg8 = np.array([[1, 1, 1],
                                  [1, 8, 1],
                                  [1, 1, 1]])

# Apply the Laplacian kernels to the image
laplacian_filtered_pos8 = cv2.filter2D(image, -1, laplacian_kernel_pos8)
laplacian_filtered_neg8 = cv2.filter2D(image, -1, laplacian_kernel_neg8)

# Enhance the images by adding the Laplacian filtered images to the original image
enhanced_image_pos8 = cv2.add(image, laplacian_filtered_pos8)
enhanced_image_neg8 = cv2.add(image, laplacian_filtered_neg8)

# Plot the original image, Laplacian filtered images, and enhanced images
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(laplacian_filtered_pos8, cmap='gray')
plt.title('Laplacian Filtered Image (+8)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(enhanced_image_pos8, cmap='gray')
plt.title('Enhanced Image (+8)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(laplacian_filtered_neg8, cmap='gray')
plt.title('Laplacian Filtered Image (-8)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(enhanced_image_neg8, cmap='gray')
plt.title('Enhanced Image (-8)')
plt.axis('off')

plt.show()
