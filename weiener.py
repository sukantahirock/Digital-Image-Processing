import cv2
import numpy as np
from matplotlib import pyplot as plt

def weiner_filter(img, K=0.01):
    # Convert image to float32 for processing
    img_float32 = np.float32(img)
    
    # Perform Fourier Transform
    f = np.fft.fft2(img_float32)
    fshift = np.fft.fftshift(f)
    
    # Estimate Power Spectra of Noise
    rows, cols = img.shape
    M, N = np.meshgrid(np.arange(cols), np.arange(rows))
    noise_power_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(np.random.randn(rows, cols))))**2
    noise_power_spectrum_mean = np.mean(noise_power_spectrum)
    
    # Estimate Power Spectra of Signal
    signal_power_spectrum = np.abs(fshift)**2
    
    # Wiener Filter
    H = np.conj(signal_power_spectrum) / (signal_power_spectrum + K * noise_power_spectrum_mean)
    filtered_fshift = H * fshift
    
    # Inverse Fourier Transform
    f_filtered = np.fft.ifftshift(filtered_fshift)
    img_filtered = np.fft.ifft2(f_filtered)
    img_filtered = np.abs(img_filtered)
    
    return img_filtered

# Read image
img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

# Apply Wiener filter
filtered_img = weiner_filter(img)

# Display original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(filtered_img, cmap='gray')
plt.title('Filtered Image')
plt.show()
