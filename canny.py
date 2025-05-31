import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('canny.png', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Canny edge detection
edges = cv2.Canny(blurred, 100, 200)  # Adjust threshold values as needed

# Display the original image and edges
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')

plt.show()
