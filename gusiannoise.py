import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('lena.png')

# Define function to apply filters and calculate metrics
def apply_filters_and_calculate_metrics(noisy_image):
    # Generate Gaussian noise
    sigma = 25  # Define your desired sigma value for Gaussian noise
    mean = 0
    var = sigma ** 2
    gaussian = np.random.normal(mean, sigma, noisy_image.shape)
    noisy_image_gaussian = np.clip(noisy_image + gaussian, 0, 255).astype(np.uint8)

    # Apply filters
    average_filter = cv2.blur(noisy_image_gaussian, (3, 3))
    max_filter = cv2.dilate(noisy_image_gaussian, None)
    min_filter = cv2.erode(noisy_image_gaussian, None)
    laplacian_filter = cv2.Laplacian(noisy_image_gaussian, cv2.CV_64F)
    bilateral_filter = cv2.bilateralFilter(noisy_image_gaussian, 9, 75, 75)

    # Calculate PSNR, SNR, MSE, EPF
    def calculate_metrics(original, filtered):
        mse = np.mean((original - filtered) ** 2)
        snr = 10 * np.log10(np.mean(original ** 2) / mse)
        psnr = 10 * np.log10(255 ** 2 / mse)
        epf = 100 * (1 - np.sqrt(mse) / 255)
        return mse, snr, psnr, epf

    mse_avg, snr_avg, psnr_avg, epf_avg = calculate_metrics(image, average_filter)
    mse_max, snr_max, psnr_max, epf_max = calculate_metrics(image, max_filter)
    mse_min, snr_min, psnr_min, epf_min = calculate_metrics(image, min_filter)
    mse_lap, snr_lap, psnr_lap, epf_lap = calculate_metrics(image, laplacian_filter)
    mse_bil, snr_bil, psnr_bil, epf_bil = calculate_metrics(image, bilateral_filter)

    return (mse_avg, snr_avg, psnr_avg, epf_avg), \
           (mse_max, snr_max, psnr_max, epf_max), \
           (mse_min, snr_min, psnr_min, epf_min), \
           (mse_lap, snr_lap, psnr_lap, epf_lap), \
           (mse_bil, snr_bil, psnr_bil, epf_bil)

# Apply filters and calculate metrics
metrics = apply_filters_and_calculate_metrics(image)

# Plotting
labels = ['Average Filter', 'Max Filter', 'Min Filter', 'Laplacian Filter', 'Bilateral Filter']
metrics_names = ['MSE', 'SNR (dB)', 'PSNR (dB)', 'EPF (%)']
colors = ['b', 'g', 'r', 'c']

plt.figure(figsize=(12, 6))
for i in range(len(metrics)):
    plt.subplot(2, 3, i + 1)
    plt.bar(metrics_names, metrics[i], color=colors)
    plt.title(labels[i])
    plt.xlabel('Metrics')
    plt.ylabel('Values')

plt.tight_layout()
plt.show()
