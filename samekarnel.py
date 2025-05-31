import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('avarage.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define a kernel (e.g., Laplacian)
kernel = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]])

# Apply the kernel to the image multiple times
num_iterations = 5
filtered_images = []
filtered_image = gray_image.copy()
for i in range(num_iterations):
    filtered_image = cv2.filter2D(filtered_image, -1, kernel)
    filtered_images.append(filtered_image)

# Plot the original and filtered images using subplots
plt.figure(figsize=(15, 5))

plt.subplot(1, num_iterations + 1, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

for i, filtered_image in enumerate(filtered_images):
    plt.subplot(1, num_iterations + 1, i + 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'Iteration {i + 1}')
    plt.axis('off')

plt.show()
