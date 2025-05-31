import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image in grayscale
input_image = cv2.imread('spine.png', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if input_image is None:
    print("Error: Could not open or read the image.")
    exit()

# Set the intensity range for slicing
low_intensity = 20
high_intensity = 220

# Create a binary mask based on the intensity range
binary_mask = np.zeros_like(input_image)
binary_mask[(input_image >= low_intensity) & (input_image <= high_intensity)] = 255

# Apply the binary mask to the original image
sliced_image = cv2.bitwise_and(input_image, binary_mask)

# Display the images using Matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(binary_mask, cmap='gray')
plt.title('Binary Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sliced_image, cmap='gray')
plt.title('Gray Level Slicing')
plt.axis('off')

plt.show()
