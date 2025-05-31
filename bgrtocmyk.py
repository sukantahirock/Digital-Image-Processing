import cv2
import numpy as np
import matplotlib.pyplot as plt

def bgr_to_cmyk(bgr):
    b, g, r = bgr
    c = 1 - (r / 255)
    m = 1 - (g / 255)
    y = 1 - (b / 255)
    k = min(c, m, y)
    if k == 1:
        return 0, 0, 0, 1
    return (
        round((c - k) / (1 - k), 3),
        round((m - k) / (1 - k), 3),
        round((y - k) / (1 - k), 3),
        round(k, 3),
    )

# Load the image
img = cv2.imread("fruits.jpg")

# Convert the image from BGR to CMYK
cmyk_img = np.apply_along_axis(bgr_to_cmyk, axis=-1, arr=img)

# Display the original and converted image side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original Image")
axs[0].axis("off")
axs[1].imshow(cmyk_img)
axs[1].set_title("CMYK Image")
axs[1].axis("off")
plt.show()
