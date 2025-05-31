import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('cameraman.jpeg', cv2.IMREAD_GRAYSCALE)

# Define the Laplacian kernel for +4 and -4
laplacian_kernel_pos = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])

laplacian_kernel_neg = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]])

# Apply the Laplacian kernels to the image
laplacian_filtered_pos = cv2.filter2D(image, -1, laplacian_kernel_pos)
laplacian_filtered_neg = cv2.filter2D(image, -1, laplacian_kernel_neg)

# Enhance the images by adding the Laplacian filtered images to the original image
enhanced_image_pos = cv2.add(image, laplacian_filtered_pos)
enhanced_image_neg = cv2.add(image, laplacian_filtered_neg)

# Plot the original image, Laplacian filtered images, and enhanced images
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(laplacian_filtered_pos, cmap='gray')
plt.title('Laplacian Filtered Image (+4)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(enhanced_image_pos, cmap='gray')
plt.title('Enhanced Image (+4)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(laplacian_filtered_neg, cmap='gray')
plt.title('Laplacian Filtered Image (-4)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(enhanced_image_neg, cmap='gray')
plt.title('Enhanced Image (-4)')
plt.axis('off')

plt.show()
