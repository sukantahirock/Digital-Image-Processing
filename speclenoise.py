import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('lena.png')

# Set sigma values
sigmas = [0.01, 0.05, 0.1]  # Adjust sigma values as needed

# Apply filters and calculate metrics
def apply_filters_and_calculate_metrics(noisy_image):
    average_filter = cv2.blur(noisy_image, (3, 3))
    max_filter = cv2.dilate(noisy_image, None)
    min_filter = cv2.erode(noisy_image, None)
    bilateral_filter = cv2.bilateralFilter(noisy_image, 9, 75, 75)  # Bilateral filter

    # Calculate MSE, SNR, PSNR
    def calculate_metrics(original, filtered):
        mse = np.mean((original - filtered) ** 2)
        snr = 10 * np.log10(np.mean(original ** 2) / mse)
        psnr = 10 * np.log10(255 ** 2 / mse)
        return mse, snr, psnr

    mse_avg, snr_avg, psnr_avg = calculate_metrics(image, average_filter)
    mse_max, snr_max, psnr_max = calculate_metrics(image, max_filter)
    mse_min, snr_min, psnr_min = calculate_metrics(image, min_filter)
    mse_bilateral, snr_bilateral, psnr_bilateral = calculate_metrics(image, bilateral_filter)  # Bilateral metrics

    return (average_filter, mse_avg, snr_avg, psnr_avg), \
           (max_filter, mse_max, snr_max, psnr_max), \
           (min_filter, mse_min, snr_min, psnr_min), \
           (bilateral_filter, mse_bilateral, snr_bilateral, psnr_bilateral)  # Return metrics for bilateral filter

plt.figure(figsize=(18, 10))

for i, sigma in enumerate(sigmas, start=1):
    # Add speckle noise with the current sigma value
    noise = sigma * np.random.randn(*image.shape)
    noisy_image = np.clip(image + image * noise, 0, 255).astype(np.uint8)

    # Apply filters and calculate metrics for the current noisy image
    (avg, maxf, minf, bilateral) = apply_filters_and_calculate_metrics(noisy_image)  # Include bilateral filter metrics

    # Plot noisy image and filtered images
    plt.subplot(3, 5, (i-1)*5 + 1)
    plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Sigma: {sigma}\nSpeckle Noisy Image')
    plt.axis('off')

    plt.subplot(3, 5, (i-1)*5 + 2)
    plt.imshow(cv2.cvtColor(avg[0], cv2.COLOR_BGR2RGB))
    plt.title(f'Average Filter\nMSE: {avg[1]:.2f}\nSNR: {avg[2]:.2f} dB\nPSNR: {avg[3]:.2f} dB')
    plt.axis('off')

    plt.subplot(3, 5, (i-1)*5 + 3)
    plt.imshow(cv2.cvtColor(maxf[0], cv2.COLOR_BGR2RGB))
    plt.title(f'Max Filter\nMSE: {maxf[1]:.2f}\nSNR: {maxf[2]:.2f} dB\nPSNR: {maxf[3]:.2f} dB')
    plt.axis('off')

    plt.subplot(3, 5, (i-1)*5 + 4)
    plt.imshow(cv2.cvtColor(minf[0], cv2.COLOR_BGR2RGB))
    plt.title(f'Min Filter\nMSE: {minf[1]:.2f}\nSNR: {minf[2]:.2f} dB\nPSNR: {minf[3]:.2f} dB')
    plt.axis('off')

    plt.subplot(3, 5, (i-1)*5 + 5)
    plt.imshow(cv2.cvtColor(bilateral[0], cv2.COLOR_BGR2RGB))  # Plot bilateral filtered image
    plt.title(f'Bilateral Filter\nMSE: {bilateral[1]:.2f}\nSNR: {bilateral[2]:.2f} dB\nPSNR: {bilateral[3]:.2f} dB')
    plt.axis('off')

plt.tight_layout()
plt.show()
