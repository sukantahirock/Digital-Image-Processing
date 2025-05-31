
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('region.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary mask
_, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

# Copy the mask to ensure the original image doesn't get overwritten
filled_image = mask.copy()

# Choose a seed point (x, y) within the region you want to fill
seed_point = (100, 100)

# Perform region filling using flood fill algorithm
cv2.floodFill(filled_image, None, seed_point, 255)

# Invert the filled image (optional)
filled_image = cv2.bitwise_not(filled_image)

# Display the original image and the filled region using subplots
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filled_image, cmap='gray')
plt.title('Filled Region')
plt.axis('off')

plt.show()

