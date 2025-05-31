import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
image1 = cv2.imread('add1.png')
image2 = cv2.imread('add2.png')

# Perform addition
add = cv2.add(image1, image2)
add2 =image1+50

# Display the images
plt.subplot(2, 2, 1)
plt.imshow(image1)
plt.title('Image 1')

plt.subplot(2, 2, 2)
plt.imshow(image2)
plt.title('Image 2')

plt.subplot(2, 2, 3)
plt.imshow(add)
plt.title('Addition')
plt.subplot(2, 2, 4)
plt.imshow(add2)
plt.title('image+50')
plt.show()

