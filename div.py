import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
image1 = cv2.imread('div2.png')
image2 = cv2.imread('div1.png')

# Resize images to have the same dimensions
height, width = min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1])
image1 = cv2.resize(image1, (width, height))
image2 = cv2.resize(image2, (width, height))

# Perform division
divide = cv2.divide(image1.astype(np.float32), image2.astype(np.float32))

# Display the images
plt.subplot(1, 3, 1)
plt.imshow(image1)
plt.title('Image 1')

plt.subplot(1, 3, 2)
plt.imshow(image2)
plt.title('Image 2')

plt.subplot(1, 3, 3)
plt.imshow(divide)
plt.title('Division')
plt.show()
