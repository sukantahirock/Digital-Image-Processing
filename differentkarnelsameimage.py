import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_snr(original_image, noisy_image):
    signal = np.sum(original_image ** 2) # Corrected
    noise = np.sum((original_image - noisy_image) ** 2) # Corrected
    snr = 10 * np.log10(signal / noise)
    return snr

def calculate_psnr(original_image, noisy_image):
    mse = np.mean((original_image - noisy_image) ** 2) # Corrected
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    return psnr

def calculate_msc(original_image, noisy_image):
    msc = np.mean((original_image - noisy_image) ** 2) # Corrected
    return msc

# Load the image
image = cv2.imread('lena.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define a kernel (e.g., Laplacian)
kernel = np.array([[0, -1, 0],
                   [-1, 4, -1],  # Corrected
                   [0, -1, 0]])

# Apply the kernel to the image multiple times
num_iterations = 5
filtered_images = []
metrics = []

filtered_image = gray_image.copy()
for i in range(num_iterations):
    filtered_image = cv2.filter2D(filtered_image, -1, kernel)
    filtered_images.append(filtered_image)
    snr = calculate_snr(gray_image, filtered_image)
    psnr = calculate_psnr(gray_image, filtered_image)
    msc = calculate_msc(gray_image, filtered_image)
    metrics.append({'SNR': snr, 'PSNR': psnr, 'MSC': msc})

# Plot the original and filtered images using subplots
plt.figure(figsize=(15, 5))

plt.subplot(1, num_iterations + 1, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

for i, (filtered_image, metric) in enumerate(zip(filtered_images, metrics)):
    plt.subplot(1, num_iterations + 1, i + 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'Iteration {i + 1}\n SNR: {metric["SNR"]:.2f}\n PSNR: {metric["PSNR"]:.2f}\n MSC: {metric["MSC"]:.2f}') # Corrected
    plt.axis('off')

plt.show()
