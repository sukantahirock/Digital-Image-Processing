import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('median.png', cv2.IMREAD_GRAYSCALE)

# Get image dimensions
h, w = img.shape

# Median Filtering Function
def median_filter(img, k):
    pad = k // 2
    img_new = np.zeros_like(img)
    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            # Extract the neighborhood
            neighborhood = img[i-pad:i+pad+1, j-pad:j+pad+1]
            # Apply median operation
            img_new[i, j] = np.median(neighborhood)
    return img_new

# Apply Median Filter
img_median = median_filter(img, 3)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(img_median, cmap='gray')
axes[1].set_title('Median Filter')
axes[1].axis('off')

plt.show()
