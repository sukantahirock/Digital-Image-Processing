

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('einstine.jpg',0)
hist1 = cv2.calcHist([img],[0],None,[256],[0,256])
img_2 = cv2.equalizeHist(img)
hist2 = cv2.calcHist([img_2],[0],None,[256],[0,256])
plt.subplot(221),plt.imshow(img);
plt.subplot(222),plt.plot(hist1);
plt.subplot(223),plt.imshow(img_2);
plt.subplot(224),plt.plot(hist2);