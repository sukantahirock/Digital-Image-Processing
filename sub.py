import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
image1 = cv2.imread('elegira.jpg')
image2 = cv2.imread('elephant.jpg')


# Perform subtraction
subtract = cv2.subtract(image1, image2)

# Display the images
plt.subplot(1, 3, 1)
plt.imshow(image1)
plt.title('Image 1')

plt.subplot(1, 3, 2)
plt.imshow(image2)
plt.title('Image 2')

plt.subplot(1, 3, 3)
plt.imshow(subtract)
plt.title('Subtraction')
plt.show()
