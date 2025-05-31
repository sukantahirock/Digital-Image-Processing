import numpy as np
import matplotlib.pyplot as plt

# Function to apply average filter to an image
def average_filter(image, kernel_size, n):
    # Get image dimensions
    height, width = image.shape
    
    # Initialize filtered image
    filtered_image = np.copy(image)
    
    # Plot original image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, n+1, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Apply average filtering n times
    for i in range(n):
        filtered_image_temp = np.zeros_like(image)
        for j in range(height):
            for k in range(width):
                # Define region of interest
                top = max(0, j - kernel_size // 2)
                bottom = min(height - 1, j + kernel_size // 2)
                left = max(0, k - kernel_size // 2)
                right = min(width - 1, k + kernel_size // 2)
                
                # Compute average value in the region of interest
                region = filtered_image[top:bottom+1, left:right+1]
                filtered_image_temp[j, k] = np.mean(region)
        
        # Update filtered image
        filtered_image = np.copy(filtered_image_temp)
        
        # Plot filtered image
        plt.subplot(1, n+1, i+2)
        plt.imshow(filtered_image, cmap='gray')
        plt.title('Iteration {}'.format(i+1))
        plt.axis('off')
    
    plt.show()

# Load an example image (you can use any image you want)
# Replace 'example_image.jpg' with the path to your image
image = plt.imread('avaragen.png')

# Convert image to grayscale if it's RGB
if len(image.shape) == 3:
    image = np.mean(image, axis=2)

# Prompt user to input kernel size and number of iterations
kernel_size = int(input("Enter kernel size: "))
n_iterations = int(input("Enter number of iterations: "))

# Apply average filtering with the specified kernel size and number of iterations
average_filter(image, kernel_size, n_iterations)
