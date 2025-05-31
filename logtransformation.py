import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

# Perform log transformation
c = 255 / np.log(1 + np.max(image))
log_transformed = c * np.log(1 + image)

# Convert to uint8
log_transformed = np.uint8(log_transformed)

# Plot the original and transformed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(log_transformed, cmap='gray')
plt.title('Log Transformed Image')

plt.show()
