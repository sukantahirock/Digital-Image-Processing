import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the RGB image using OpenCV
image = cv2.imread('circlehole.png')

# Split the image into its RGB channels
b, g, r = cv2.split(image)

# Define a kernel for dilation
kernel = np.ones((5, 5), np.uint8)  # You can adjust the kernel size as needed

# Perform multiple dilations and visualize the result after each iteration
num_iterations =5
plt.figure(figsize=(15, 6))

for i in range(num_iterations):
    # Perform dilation on each channel
    dilated_b = cv2.dilate(b, kernel, iterations=i+1)
    dilated_g = cv2.dilate(g, kernel, iterations=i+1)
    dilated_r = cv2.dilate(r, kernel, iterations=i+1)

    # Merge the dilated channels back into an RGB image
    dilated_image = cv2.merge((dilated_b, dilated_g, dilated_r))

    # Plot the dilated image
    plt.subplot(2, 5, i+1)
    plt.imshow(cv2.cvtColor(dilated_image, cv2.COLOR_BGR2RGB))
    plt.title('Dilation {}'.format(i+1))
    plt.axis('off')

plt.tight_layout()
plt.show()
