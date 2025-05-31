import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the image
image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)


# Define the coordinates of the point
x, y = 100, 100  # Adjust these coordinates as needed

# Define the Laplacian kernels
laplacian_kernel1 = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]])

laplacian_kernel2 = np.array([[1, 1, 1],
                              [1, -8, 1],
                              [1, 1, 1]])

# Extract the neighborhood around the point
neighborhood = image[y-1:y+2, x-1:x+2]

# Apply the Laplacian kernels to the neighborhood
result1 = np.sum(neighborhood * laplacian_kernel1)
result2 = np.sum(neighborhood * laplacian_kernel2)

# Create a copy of the original image
result_image = image.copy()

# Update the pixel value at the specified point with the result of applying the kernels
result_image[y, x] = np.clip(result1, 0, 255)  # Ensure the pixel value is within [0, 255]
result_image[y, x] = np.clip(result2, 0, 255)  # Ensure the pixel value is within [0, 255]

# Plot the original image, resultant image, and Laplacian kernels
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(result_image, cmap='gray')
plt.title('Resultant Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(laplacian_kernel1), cmap='gray')
plt.title('Laplacian Kernel 1')
plt.axis('off')

plt.show()

print("Result using Laplacian kernel 1:", result1)
print("Laplacian Kernel 1:")
print(laplacian_kernel1)

print("\nResult using Laplacian kernel 2:", result2)
print("Laplacian Kernel 2:")
print(laplacian_kernel2)