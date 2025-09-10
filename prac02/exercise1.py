'''
Machine Perception Prac02
Daehwan Yeo  

Exercise 1 - Colour Conversion

To see the raw channels of HSV / Luv / Lab → use img_hsv, img_luv, img_lab
To see the converted image as humans would see → use hsv_disp, luv_disp, lab_disp
in imshow().
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in BGR format
image_path = 'prac02ex01img01.jpg'
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Cannot load image: {image_path}")

# Convert BGR to RGB (for correct display in matplotlib)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Convert BGR to grayscale
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Convert BGR to HSV, Luv, Lab
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_luv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Luv)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)

# Convert them back to RGB for visualization
hsv_disp = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
luv_disp = cv2.cvtColor(img_luv, cv2.COLOR_Luv2RGB)
lab_disp = cv2.cvtColor(img_lab, cv2.COLOR_Lab2RGB)

# Plot images using subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Display grayscale
axs[0, 0].imshow(img_gray, cmap='gray')
axs[0, 0].set_title('Grayscale Image')

# Display HSV
axs[0, 1].imshow(img_hsv)
axs[0, 1].set_title('HSV Image')

# Display Luv
axs[1, 0].imshow(img_luv)
axs[1, 0].set_title('Luv Image')

# Display Lab
axs[1, 1].imshow(img_lab)
axs[1, 1].set_title('Lab Image')

# Remove axis for all subplots
for ax in axs.flat:
    ax.axis('off')

# Save the figure
output_filename = "ex01_result.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_filename}")
