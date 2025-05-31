import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('lena.png')

# Ask user for sigma value
sigma = float(input("Enter the sigma value for Gaussian noise: "))

# Add Gaussian noise
mean = 0
var = sigma ** 2
gaussian = np.random.normal(mean, sigma, image.shape)
noisy_image_gaussian = np.clip(image + gaussian, 0, 255).astype(np.uint8)

# Add Speckle noise
speckle = np.random.normal(0, 0.1, image.shape)
noisy_image_speckle = np.clip(image + image * speckle, 0, 255).astype(np.uint8)

# Apply filters and calculate metrics
def apply_filters_and_calculate_metrics(noisy_image):
    average_filter = cv2.blur(noisy_image, (3, 3))
    max_filter = cv2.dilate(noisy_image, None)
    min_filter = cv2.erode(noisy_image, None)
    laplacian_filter = cv2.Laplacian(noisy_image, cv2.CV_64F)
    bilateral_filter = cv2.bilateralFilter(noisy_image, 9, 75, 75)

    # Calculate MSE, SNR, PSNR
    def calculate_metrics(original, filtered):
        mse = np.mean((original - filtered) ** 2)
        snr = 10 * np.log10(np.mean(original ** 2) / mse)
        psnr = 10 * np.log10(255 ** 2 / mse)
        return mse, snr, psnr

    mse_avg, snr_avg, psnr_avg = calculate_metrics(image, average_filter)
    mse_max, snr_max, psnr_max = calculate_metrics(image, max_filter)
    mse_min, snr_min, psnr_min = calculate_metrics(image, min_filter)
    mse_lap, snr_lap, psnr_lap = calculate_metrics(image, laplacian_filter)
    mse_bil, snr_bil, psnr_bil = calculate_metrics(image, bilateral_filter)

    return (average_filter, mse_avg, snr_avg, psnr_avg), \
           (max_filter, mse_max, snr_max, psnr_max), \
           (min_filter, mse_min, snr_min, psnr_min), \
           (laplacian_filter, mse_lap, snr_lap, psnr_lap), \
           (bilateral_filter, mse_bil, snr_bil, psnr_bil)

# Apply filters and calculate metrics for both noisy images
(gaussian_avg, gaussian_max, gaussian_min, gaussian_lap, gaussian_bil), \
(speckle_avg, speckle_max, speckle_min, speckle_lap, speckle_bil) = \
    apply_filters_and_calculate_metrics(noisy_image_gaussian), \
    apply_filters_and_calculate_metrics(noisy_image_speckle)

# Display results using matplotlib
plt.figure(figsize=(18, 10))

# Gaussian Noise Results
plt.subplot(2, 5, 1)
plt.imshow(cv2.cvtColor(noisy_image_gaussian, cv2.COLOR_BGR2RGB))
plt.title('Gaussian Noisy Image')
plt.axis('off')

plt.subplot(2, 5, 2)
plt.imshow(cv2.cvtColor(gaussian_avg[0], cv2.COLOR_BGR2RGB))
plt.title(f'Average Filter\nMSE: {gaussian_avg[1]:.2f}\nSNR: {gaussian_avg[2]:.2f} dB\nPSNR: {gaussian_avg[3]:.2f} dB')
plt.axis('off')

plt.subplot(2, 5, 3)
plt.imshow(cv2.cvtColor(gaussian_max[0], cv2.COLOR_BGR2RGB))
plt.title(f'Max Filter\nMSE: {gaussian_max[1]:.2f}\nSNR: {gaussian_max[2]:.2f} dB\nPSNR: {gaussian_max[3]:.2f} dB')
plt.axis('off')

plt.subplot(2, 5, 4)
plt.imshow(cv2.cvtColor(gaussian_min[0], cv2.COLOR_BGR2RGB))
plt.title(f'Min Filter\nMSE: {gaussian_min[1]:.2f}\nSNR: {gaussian_min[2]:.2f} dB\nPSNR: {gaussian_min[3]:.2f} dB')
plt.axis('off')

plt.subplot(2, 5, 5)
plt.imshow(cv2.cvtColor(gaussian_bil[0], cv2.COLOR_BGR2RGB))
plt.title(f'Bilateral Filter\nMSE: {gaussian_bil[1]:.2f}\nSNR: {gaussian_bil[2]:.2f} dB\nPSNR: {gaussian_bil[3]:.2f} dB')
plt.axis('off')

# Speckle Noise Results
plt.subplot(2, 5, 6)
plt.imshow(cv2.cvtColor(noisy_image_speckle, cv2.COLOR_BGR2RGB))
plt.title('Speckle Noisy Image')
plt.axis('off')

plt.subplot(2, 5, 7)
plt.imshow(cv2.cvtColor(speckle_avg[0], cv2.COLOR_BGR2RGB))
plt.title(f'Average Filter\nMSE: {speckle_avg[1]:.2f}\nSNR: {speckle_avg[2]:.2f} dB\nPSNR: {speckle_avg[3]:.2f} dB')
plt.axis('off')

plt.subplot(2, 5, 8)
plt.imshow(cv2.cvtColor(speckle_max[0], cv2.COLOR_BGR2RGB))
plt.title(f'Max Filter\nMSE: {speckle_max[1]:.2f}\nSNR: {speckle_max[2]:.2f} dB\nPSNR: {speckle_max[3]:.2f} dB')
plt.axis('off')

plt.subplot(2, 5, 9)
plt.imshow(cv2.cvtColor(speckle_min[0], cv2.COLOR_BGR2RGB))
plt.title(f'Min Filter\nMSE: {speckle_min[1]:.2f}\nSNR: {speckle_min[2]:.2f} dB\nPSNR: {speckle_min[3]:.2f} dB')
plt.axis('off')

plt.subplot(2, 5, 10)
plt.imshow(cv2.cvtColor(speckle_bil[0], cv2.COLOR_BGR2RGB))
plt.title(f'Bilateral Filter\nMSE: {speckle_bil[1]:.2f}\nSNR: {speckle_bil[2]:.2f} dB\nPSNR: {speckle_bil[3]:.2f} dB')
plt.axis('off')

plt.show()
