
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('lena.png',0)

# Define the Laplacian kernels
laplacian_kernel1 = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]])

laplacian_kernel2 = np.array([[1, 1, 1],
                              [1, -8, 1],
                              [1, 1, 1]])

# Apply the Laplacian kernels to the full image
result1 = cv2.filter2D(image, -1, laplacian_kernel1)
result2 = cv2.filter2D(image, -1, laplacian_kernel2)

# Plot the original image and resultant images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(result1, cmap='gray')
plt.title('Result using Laplacian kernel 1')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(result2, cmap='gray')
plt.title('Result using Laplacian kernel 2')
plt.axis('off')

plt.show()
