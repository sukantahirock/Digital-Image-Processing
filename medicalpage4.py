import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
image_a = cv2.imread('medicalpage4a.png', cv2.IMREAD_GRAYSCALE)
image_b = cv2.imread('medicalpage4b.png', cv2.IMREAD_GRAYSCALE)

# Ensure both images have the same dimensions
height = min(image_a.shape[0], image_b.shape[0])
width = min(image_a.shape[1], image_b.shape[1])
image_a_resized = cv2.resize(image_a, (width, height))
image_b_resized = cv2.resize(image_b, (width, height))

# Compute the difference A - B
difference_ab = np.abs(image_a_resized - image_b_resized)

# Compute the difference B - A
difference_ba = np.abs(image_b_resized - image_a_resized)

# Display the results
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(image_a, cmap='gray')
plt.title('Standard x-ray ILO 0/0 with added change')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(image_b, cmap='gray')
plt.title('unchanged x-ray ILO 0/0')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(difference_ab, cmap='gray')
plt.title('Absolute Difference A - B')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(difference_ba, cmap='gray')
plt.title('Absolute Difference B - A')
plt.axis('off')

plt.show()
