import cv2
import matplotlib.pyplot as plt

# Load the input image
image1 = cv2.imread('lena1.jpg')

# Convert the image to RGB format
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

# Display the images side by side using Matplotlib
plt.subplot(1, 2, 1)
plt.imshow(image1)
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(image1_rgb)
plt.title('Converted (RGB)')
plt.show()
