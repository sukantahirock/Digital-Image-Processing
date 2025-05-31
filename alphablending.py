import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the apple and orange images
apple = cv2.imread('apple.jpg')
orange = cv2.imread('orange.jpg')

# Resize images to have the same dimensions (optional)
apple = cv2.resize(apple, (orange.shape[1], orange.shape[0]))

# Define the alpha blending parameter (alpha value between 0 and 1)
alpha = 0.4

# Perform alpha blending
blended_image = cv2.addWeighted(apple, alpha, orange, 1 - alpha, 0)

# Convert BGR to RGB for Matplotlib display
apple_rgb = cv2.cvtColor(apple, cv2.COLOR_BGR2RGB)
orange_rgb = cv2.cvtColor(orange, cv2.COLOR_BGR2RGB)
blended_image_rgb = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)

# Display the images using Matplotlib
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(apple_rgb)
plt.title('Apple')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(orange_rgb)
plt.title('Orange')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(blended_image_rgb)
plt.title('Alpha Blending')
plt.axis('off')

plt.show()
