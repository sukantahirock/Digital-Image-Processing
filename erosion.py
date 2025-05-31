import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the RGB image using OpenCV
image = cv2.imread('box.png')

# Convert BGR image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Split the image into its RGB channels
r, g, b = cv2.split(image_rgb)

# Define a kernel for erosion
kernel = np.ones((9, 9), np.uint8)  # You can adjust the kernel size as needed

# Perform multiple erosions and visualize the result after each iteration
num_iterations = 5
plt.figure(figsize=(15, 6))

for i in range(num_iterations):
    # Perform erosion on each channel
    eroded_r = cv2.erode(r, kernel, iterations=i+1)
    eroded_g = cv2.erode(g, kernel, iterations=i+1)
    eroded_b = cv2.erode(b, kernel, iterations=i+1)

    # Merge the eroded channels back into an RGB image
    eroded_image = cv2.merge((eroded_r, eroded_g, eroded_b))

    # Normalize pixel values to [0, 255]
    eroded_image = np.clip(eroded_image, 0, 255).astype(np.uint8)

    # Plot the eroded image
    plt.subplot(3, 5, i+1)
    plt.imshow(eroded_image)
    plt.title('Erosion {}'.format(i+1))
    plt.axis('off')

    # Change background to red
    plt.gca().set_facecolor('red')

plt.tight_layout()
plt.show()
