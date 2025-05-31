import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the image
image = cv2.imread('lena.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define a 3x3 kernel (identity kernel)
kernel = np.array([[1, 0, 3],
                   [0, 2, 0],
                   [0, 0, 1]])

# Apply the kernel to the image
filtered_image = cv2.filter2D(gray_image, -1, kernel)

# Plot the original and filtered images using subplots
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')

plt.show()