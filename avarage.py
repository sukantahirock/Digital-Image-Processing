import numpy as np
import matplotlib.pyplot as plt

# Function to apply average filter to an image
def average_filter(image, kernel_size):
    # Get image dimensions
    height, width = image.shape
    
    # Initialize filtered image
    filtered_image = np.zeros_like(image)
    
    # Iterate over each pixel in the image
    for i in range(height):
        for j in range(width):
            # Define region of interest
            top = max(0, i - kernel_size // 2)
            bottom = min(height - 1, i + kernel_size // 2)
            left = max(0, j - kernel_size // 2)
            right = min(width - 1, j + kernel_size // 2)
            
            # Compute average value in the region of interest
            region = image[top:bottom+1, left:right+1]
            filtered_image[i, j] = np.mean(region)
    
    return filtered_image

# Load an example image (you can use any image you want)
# Replace 'example_image.jpg' with the path to your image
image = plt.imread('avarage.jpg')

# Convert image to grayscale if it's RGB
if len(image.shape) == 3:
    image = np.mean(image, axis=2)

# Prompt user to input kernel size
kernel_size = int(input("Enter kernel size: "))

# Apply average filtering with the specified kernel size
filtered_image = average_filter(image, kernel_size)

# Plot original and filtered images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Average Filtered Image (Kernel Size: {})'.format(kernel_size))
plt.axis('off')

plt.show()
