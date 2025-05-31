import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lena.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = img.shape

# Average Filtering
def fun(i, j, k, mid):
    val = 0.0
    for y in range(0, k):
        for z in range(0, k):
            a = i - mid + y
            b = j - mid + z
            val += (img[a, b] * mask[y, z])
    return round(val)

fig = plt.figure()
fig.add_subplot(3, 2, 1)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title("Original Image")

g = 1
for k in range(3, 12, 2):

    mask = np.ones([k, k], dtype=int)
    mask = mask / (k * k)

    img_new = np.zeros([h, w])
    mid = int(k / 2)

    for i in range(mid, h - mid):
        for j in range(mid, w - mid):
            img_new[i, j] = fun(i, j, k, mid)

    sq_sum = 0
    abs_sum = 0
    for i in range(0, h):
        for j in range(0, w):
            sq_sum += ((img[i, j]) * (img[i, j]))
            abs_sum += (img[i, j] - img_new[i, j]) * (img[i, j] - img_new[i, j])

    # MSE
    mse = (abs_sum / (h * w))
    print("MSE of Kernel size", k, "x", k, "is", mse)

    # SNR
    snr = 20 * math.log10(sq_sum / abs_sum)
    print("SNR of Kernel size", k, "x", k, "is", snr)

    # PSNR
    psnr = 20 * math.log10(((255 * 255) * (h * w)) / abs_sum)
    print("PSNR of Kernel size", k, "x", k, "is", psnr)
    print()

    g = g + 1

    fig.add_subplot(3, 2, g)
    plt.imshow(img_new, cmap='gray')
    plt.axis('off')
    plt.title("Kernel Size: " + str(k))

print('End')
