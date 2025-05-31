import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
image1 = cv2.imread('moon.jpg')
image2 = cv2.imread('moon.jpg', cv2.IMREAD_GRAYSCALE)

# Perform multiplication
multiply = cv2.multiply(image1, 1.2)
multiply1 = cv2.multiply(image2, 1.2)

# Convert BGR image to RGB for proper display
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
multiply_rgb = cv2.cvtColor(multiply, cv2.COLOR_BGR2RGB)

# Display the images
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image1_rgb)
plt.title('Original (BGR)')

plt.subplot(2, 2, 2)
plt.imshow(multiply_rgb)
plt.title('Multiplication (BGR)')

plt.subplot(2, 2, 3)
plt.imshow(image2, cmap='gray')
plt.title('Grayscale')

plt.subplot(2, 2, 4)
plt.imshow(multiply1, cmap='gray')
plt.title('Multiplication (Grayscale)')

plt.show()
