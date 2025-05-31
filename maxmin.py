import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("min.jpg", 0)
h, w = img.shape

# Function for Maximum Filtering
def max_filter(i, j, k, mid):
    max_val = float('-inf')
    for y in range(0, k):
        for z in range(0, k):
            a = i - mid + y
            b = j - mid + z
            max_val = max(max_val, img[a, b])
    return max_val

# Function for Minimum Filtering
def min_filter(i, j, k, mid):
    min_val = float('inf')
    for y in range(0, k):
        for z in range(0, k):
            a = i - mid + y
            b = j - mid + z
            min_val = min(min_val, img[a, b])
    return min_val

fig = plt.figure()
fig.add_subplot(3, 3, 1)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title("Original Image")

# Kernel
mask = np.ones([3, 3], dtype=int)
mask = mask / 9

for filter_type, title in [('Max', 'Maximum'), ('Min', 'Minimum')]:
    for k in range(1, 3):
        img_filtered = np.zeros([h, w])

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if filter_type == 'Max':
                    img_filtered[i, j] = max_filter(i, j, 3, 1)
                elif filter_type == 'Min':
                    img_filtered[i, j] = min_filter(i, j, 3, 1)

        plt.subplot(3, 3, 3 * (k - 1) + (2 if filter_type == 'Max' else 3))
        plt.imshow(img_filtered, cmap='gray')
        plt.axis('off')
        plt.title(f"{title} {k} Times")

plt.show()
print('End')
