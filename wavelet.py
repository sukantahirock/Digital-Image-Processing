import pywt
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the grayscale image
image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

# Perform DWT
coeffs = pywt.dwt2(image, 'haar')

# Compression (Thresholding)
threshold = 20
coeffs_compressed = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

# Inverse DWT
reconstructed_image = pywt.idwt2(coeffs_compressed, 'haar')

# Extract LL, LH, HL, HH components
LL, (LH, HL, HH) = coeffs

# Display original and reconstructed images using Matplotlib
plt.figure(figsize=(14, 10))

plt.subplot(4, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(4, 2, 3)
plt.imshow(LL, cmap='gray')
plt.title('LL Component')
plt.axis('off')

plt.subplot(4, 2, 4)
plt.imshow(LH, cmap='gray')
plt.title('LH Component')
plt.axis('off')

plt.subplot(4, 2, 5)
plt.imshow(HL, cmap='gray')
plt.title('HL Component')
plt.axis('off')

plt.subplot(4, 2, 6)
plt.imshow(HH, cmap='gray')
plt.title('HH Component')
plt.axis('off')

plt.subplot(4, 2, 8)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Compressed Image')
plt.axis('off')

plt.show()
