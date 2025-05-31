import numpy as np
import cv2

# Load the image in greyscale
img = cv2.imread('clahe.jpg', 0)

# Apply histogram equalization (uncomment if needed)
# img_2 = cv2.equalizeHist(img)

# Create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
out = clahe.apply(img)

# Display the images side by side using cv2.hconcat
out1 = cv2.hconcat([img, out])
cv2.imshow('Comparison', out1)
cv2.waitKey(0)
cv2.destroyAllWindows()
