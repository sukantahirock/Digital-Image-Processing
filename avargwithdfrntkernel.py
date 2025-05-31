import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('avg.png')

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function to apply average filter
def apply_average_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

# Define kernel sizes
kernel_sizes = [3, 5, 7, 15, 35]

# Plot original image
plt.figure(figsize=(15, 5))
plt.subplot(1, 6, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Apply average filter for different kernel sizes
for i, kernel_size in enumerate(kernel_sizes):
    filtered_image = apply_average_filter(gray_image, kernel_size)
    plt.subplot(1, 6, i + 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Kernel Size: {}'.format(kernel_size))

plt.tight_layout()
plt.show()
