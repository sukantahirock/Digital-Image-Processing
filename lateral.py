
import cv2
import numpy as np
import matplotlib.pyplot as plt

def lateral_filter(image, kernel_size=3):
    # Create a kernel for lateral filtering
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

    # Apply the filter
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image

def main():
    # Load an image
    image = cv2.imread('salt.jpg')

    if image is None:
        print("Could not read the image.")
        return

    # Convert image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply lateral filter with default kernel size (3x3)
    filtered_image = lateral_filter(image_gray)

    # Display original and filtered images
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Filtered Image')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
