import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image using OpenCV
image = cv2.imread('fourier.jpg', cv2.IMREAD_GRAYSCALE)

# Perform FFT
fft_result = np.fft.fft2(image)
fft_shift = np.fft.fftshift(fft_result)

# Compute magnitude spectrum
magnitude_spectrum = 20 * np.log(np.abs(fft_shift))

# Compute phase spectrum
phase_spectrum = np.angle(fft_shift)

# Compute inverse FFT
inverse_fft_shift = np.fft.ifftshift(fft_shift)
inverse_fft = np.fft.ifft2(inverse_fft_shift)
inverse_fft = np.abs(inverse_fft)

# Plot the original image, magnitude spectrum, phase spectrum, and the inverse FFT image
plt.figure(figsize=(12, 8))

plt.subplot(221), plt.imshow(image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(phase_spectrum, cmap='gray')
plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.imshow(inverse_fft, cmap='gray')
plt.title('Inverse FFT'), plt.xticks([]), plt.yticks([])

plt.show()
