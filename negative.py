import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the original image (apple.jpg)
apple = cv2.imread('arrow.jpg')

# Convert the original image to grayscale
input_image = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)

# Calculate the negative of the image
negative_image = 255 - input_image

# Display the images using Matplotlib
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(negative_image, cmap='gray')
plt.title('Negative Image')
plt.axis('off')

plt.show()
