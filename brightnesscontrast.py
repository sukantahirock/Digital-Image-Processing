import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read the image
im = cv.imread('C:/Users/USER/.spyder-py3/lena1.png')

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(im)
plt.axis('off')


image_shape = im.shape
image_size = im.size


image_height = im.shape[0]

# Display the shape and size information
print("Image Shape:", image_shape)
print("Image Size:", image_size)
print("Image Height:", image_height)


# Convert BGR to RGB
im_rgb = cv.cvtColor(im, cv.COLOR_BGR2RGB)

# Split image channels
b, g, r = cv.split(im)


plt.subplot(1, 2, 1)
plt.imshow(im_rgb)
plt.axis('off')
plt.title("RGB Image")

plt.subplot(1, 2, 2)
plt.imshow(r, cmap='gray')
plt.axis('off')
plt.title("Red Channel")

plt.show()


brightness = np.mean(im)
contrast = np.std(im)


print("Brightness:", brightness)
print("Contrast:", contrast)