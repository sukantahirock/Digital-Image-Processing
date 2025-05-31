import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a sample grayscale image
image = np.array([[100, 120, 150],
                   [110, 130, 160],
                   [120, 140, 170]], dtype=np.uint8)

# Define the Laplacian kernels
laplacian_kernel_4 = np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]])

laplacian_kernel_8 = np.array([[1, 1, 1],
                               [1, -8, 1],
                               [1, 1, 1]])

# Apply Laplacian filtering and enhancement for both kernels
def apply_laplacian_and_enhancement(image, laplacian_kernel):
  laplacian_filtered = cv2.filter2D(image, -1, laplacian_kernel)
  enhanced_image = cv2.add(image, laplacian_filtered)
  return enhanced_image

enhanced_image_4 = apply_laplacian_and_enhancement(image.copy(), laplacian_kernel_4)
enhanced_image_8 = apply_laplacian_and_enhancement(image.copy(), laplacian_kernel_8)

# Calculate the sum of absolute differences (SAD) between the original image and enhanced images
sad_4 = np.sum(np.abs(image.astype(np.float32) - enhanced_image_4.astype(np.float32)))
sad_8 = np.sum(np.abs(image.astype(np.float32) - enhanced_image_8.astype(np.float32)))

# Plot the original image and both enhanced images
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(enhanced_image_4, cmap='gray')
plt.title('Enhanced Image (Laplacian 4)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(enhanced_image_8, cmap='gray')
plt.title('Enhanced Image (Laplacian 8)')
plt.axis('off')

# Plot histograms for original image and both enhanced images
plt.subplot(2, 3, 4)
plt.hist(image.ravel(), bins=256, color='black')
plt.title('Histogram (Original)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(2, 3, 5)
plt.hist(enhanced_image_4.ravel(), bins=256, color='black')
plt.title('Histogram (Enhanced - Laplacian 4)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(2, 3, 6)
plt.hist(enhanced_image_8.ravel(), bins=256, color='black')
plt.title('Histogram (Enhanced - Laplacian 8)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
