import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Function to compute SNR, PSNR, and MSE
def compute_metrics(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf'), 0, 0
    max_pixel = np.max(original)
    snr = 10 * np.log10(max_pixel ** 2 / mse)
    psnr = 10 * np.log10(max_pixel ** 2 / mse)
    return snr, psnr, mse

# Function to compress coefficients
def compress_coefficients(coef, threshold):
    coef[np.abs(coef) < threshold] = 0  # Set coefficients below threshold to zero
    return coef

# Function to reconstruct image from coefficients
def reconstruct_image(LL, LH, HL, HH):
    coeffs = LL, (LH, HL, HH)
    reconstructed_image = pywt.idwt2(coeffs, 'haar')  # Perform inverse DWT
    return reconstructed_image

# Read the image
image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

# Perform first level DWT
coeffs = pywt.dwt2(image, 'haar')
LL1, (LH1, HL1, HH1) = coeffs

# Plot original image and its components
plt.figure(figsize=(12, 12))
plt.subplot(4, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(4, 3, 2)
plt.imshow(LL1, cmap='gray')
plt.title('LL1')

plt.subplot(4, 3, 3)
plt.imshow(LH1, cmap='gray')
plt.title('LH1')

plt.subplot(4, 3, 4)
plt.imshow(HL1, cmap='gray')
plt.title('HL1')

plt.subplot(4, 3, 5)
plt.imshow(HH1, cmap='gray')
plt.title('HH1')

# Compress first level coefficients
threshold = 20
compressed_LL1 = compress_coefficients(LL1.copy(), threshold)
compressed_LH1 = compress_coefficients(LH1.copy(), threshold)
compressed_HL1 = compress_coefficients(HL1.copy(), threshold)
compressed_HH1 = compress_coefficients(HH1.copy(), threshold)

# Reconstruct compressed image
compressed_image1 = reconstruct_image(compressed_LL1, compressed_LH1, compressed_HL1, compressed_HH1)

# Compute metrics for the compressed image
snr1, psnr1, mse1 = compute_metrics(image, compressed_image1)

# Plot compressed image
plt.subplot(4, 3, 6)
plt.imshow(compressed_image1, cmap='gray')
plt.title(f'Compressed Image (1st Level)\nSNR: {snr1:.2f} dB, PSNR: {psnr1:.2f} dB, MSE: {mse1:.2f}')

# Perform second level DWT on LL1
coeffs2 = pywt.dwt2(compressed_LL1, 'haar')
LL2, (LH2, HL2, HH2) = coeffs2

# Plot LL2, LH2, HL2, HH2 of second level DWT
plt.subplot(4, 3, 7)
plt.imshow(LL2, cmap='gray')
plt.title('LL2')

plt.subplot(4, 3, 8)
plt.imshow(LH2, cmap='gray')
plt.title('LH2')

plt.subplot(4, 3, 9)
plt.imshow(HL2, cmap='gray')
plt.title('HL2')

plt.subplot(4, 3, 10)
plt.imshow(HH2, cmap='gray')
plt.title('HH2')

# Compress second level coefficients
threshold2 = 10
compressed_LL2 = compress_coefficients(LL2.copy(), threshold2)
compressed_LH2 = compress_coefficients(LH2.copy(), threshold2)
compressed_HL2 = compress_coefficients(HL2.copy(), threshold2)
compressed_HH2 = compress_coefficients(HH2.copy(), threshold2)

# Reconstruct compressed image from second level coefficients
compressed_image2 = reconstruct_image(compressed_LL2, compressed_LH2, compressed_HL2, compressed_HH2)

# Compute metrics for the second compressed image
snr2, psnr2, mse2 = compute_metrics(compressed_image1, compressed_image2)

# Plot compressed image from second level
plt.subplot(4, 3, 11)
plt.imshow(compressed_image2, cmap='gray')

plt.tight_layout()
plt.show()
