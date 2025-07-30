# EXERCISE 1 - Colour Conversions

import cv2
import matplotlib.pyplot as plt
import numpy as np # Although not explicitly used for conversion here, good practice to include if using other numpy operations

# Ensure you have 'prac02ex01img01.jpg' in the same directory as this script,
# or provide the full path to the image.

img_bgr = cv2.imread("prac02ex01img01.jpg")

# Check if image was loaded successfully
if img_bgr is None:
    print("Error: Could not load image 'prac02ex01img01.jpg'. Please ensure the file exists and the path is correct.")
    exit()

# Convert BGR to RGB (for consistent display with matplotlib)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Convert BGR to grayscale
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Convert BGR to HSV
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# Convert BGR to Luv
img_luv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LUV)

# Convert BGR to Lab
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

# Plot images using subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Image Color Conversions', fontsize=16) # Add a main title for the figure

# Display the original image in the first subplot
axs[0, 0].imshow(img_rgb)
axs[0, 0].set_title('Original Image (RGB)')

# Display the grayscale image in the second subplot
axs[0, 1].imshow(img_gray, cmap='gray') # Use 'gray' colormap for grayscale images
axs[0, 1].set_title('Grayscale Image')

# Display the HSV image in the third subplot
# HSV images need to be converted back to RGB for proper display by matplotlib if not using specific HSV plotting tools
# For demonstration, we'll convert back to RGB. For analysis, you'd work directly with img_hsv.
axs[0, 2].imshow(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB))
axs[0, 2].set_title('HSV Image')

# Display the Luv image in the fourth subplot
# Luv images also need conversion for display. OpenCV Luv is typically float32.
# For display, convert to uint8 and then to RGB (though direct display might not be perceptually accurate without proper scaling)
# For visualization purposes, converting back to BGR and then to RGB for matplotlib might be done,
# but for true representation, you'd visualize channels separately or use specialized tools.
# For simple display, treating it like a normal image (which might not look right) or converting to RGB for display.
# Let's convert back to RGB for display purposes, understanding it's an approximation.
axs[1, 0].imshow(cv2.cvtColor(img_luv, cv2.COLOR_LUV2RGB))
axs[1, 0].set_title('Luv Image')

# Display the Lab image in the fifth subplot
# Similar to Luv, Lab often needs conversion for standard display.
axs[1, 1].imshow(cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB))
axs[1, 1].set_title('Lab Image')

# Fill the last subplot with nothing or a placeholder if desired
axs[1, 2].set_visible(False) # Hide the last empty subplot if not used

# Remove axis for all subplots
for ax in axs.flat:
    ax.axis('off')

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle

# Display the plot
# plt.show()
# Instead of plt.show(), save the figure to a file
output_filename = "color_conversions_plot.png" # You can choose any filename and format (e.g., .jpg, .pdf)
plt.savefig(output_filename, dpi=300, bbox_inches='tight') # dpi for resolution, bbox_inches for tight layout
print(f"Plot saved to {output_filename}")
plt.close(fig) # Close the figure to free up memory (important in scripts)