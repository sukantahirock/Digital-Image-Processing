import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the RGB image using OpenCV
image = cv2.imread('openclose.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresholded_image = cv2.threshold(gray_image, 130, 220, cv2.THRESH_BINARY)

# Define a kernel for opening and closing
kernel = np.ones((5, 5), np.uint8)  # You can adjust the kernel size as needed

# Perform opening operation
opening_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)

# Perform closing operation
closing_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)

# Perform opening and closing operations together
opening_closing_image = cv2.morphologyEx(opening_image, cv2.MORPH_CLOSE, kernel)

# Plot the original, thresholded, opened, closed, and opened-closed images
plt.figure(figsize=(20, 5))

plt.subplot(1, 5, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 5, 2)
plt.imshow(thresholded_image, cmap='gray')
plt.title('Thresholded Image')
plt.axis('off')

plt.subplot(1, 5, 3)
plt.imshow(opening_image, cmap='gray')
plt.title('Opening')
plt.axis('off')

plt.subplot(1, 5, 4)
plt.imshow(closing_image, cmap='gray')
plt.title('Closing')
plt.axis('off')

plt.subplot(1, 5, 5)
plt.imshow(opening_closing_image, cmap='gray')
plt.title('Opening + Closing')
plt.axis('off')

plt.show()
