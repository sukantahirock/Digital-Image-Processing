import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('man.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform erosion
kernel = np.ones((5, 5), np.uint8)
eroded_image = cv2.erode(gray_image, kernel, iterations=1)



# Compute the boundary image by subtracting the dilated image from the eroded image
boundary_image = gray_image-eroded_image
# Plot the original image and the boundary image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(boundary_image, cmap='gray')
plt.title('Boundary Image')
plt.axis('off')

plt.show()
