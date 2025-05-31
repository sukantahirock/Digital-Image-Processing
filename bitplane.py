import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('ful.jpg', cv2.IMREAD_GRAYSCALE)

# Extract the bit planes
bit_plane_2 = np.uint8((image & 0b00000010) >> 1) * 255
bit_plane_4 = np.uint8((image & 0b00001000) >> 3) * 255
bit_plane_6 = np.uint8((image & 0b01000000) >> 5) * 255
bit_plane_8 = np.uint8((image & 0b10000000) >> 7) * 255
bit_plane_10 = np.uint8((image & 0b00000001)) * 255  # 10th bit plane

# Display the original image and bit planes
plt.figure(figsize=(15, 15))

plt.subplot(3, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(3, 2, 2)
plt.imshow(bit_plane_2, cmap='gray')
plt.title('Bit Plane 2')
plt.axis('off')

plt.subplot(3, 2, 3)
plt.imshow(bit_plane_4, cmap='gray')
plt.title('Bit Plane 4')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.imshow(bit_plane_6, cmap='gray')
plt.title('Bit Plane 6')
plt.axis('off')

plt.subplot(3, 2, 5)
plt.imshow(bit_plane_8, cmap='gray')
plt.title('Bit Plane 8')
plt.axis('off')

plt.subplot(3, 2, 6)
plt.imshow(bit_plane_10, cmap='gray')
plt.title('Bit Plane 10')
plt.axis('off')

plt.show()