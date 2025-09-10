"""
Machine Perception Prac02
Daehwan Yeo

Exercise 5 - Histogram Equalization

Histogram equalization is a method in image processing to enhance the contrast of an image 
by redistributing the intensity levels across the entire range. 

OpenCV's equalizeHist function to perform histogram equalization on a given image.

- The equalized image has noticeably better contrast.
- The original histogram might be clustered (low dynamic range).
- The equalized histogram is spread more uniformly.
- The CDF after equalization is closer to a straight diagonal (good distribution).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------- Load grayscale image --------
image = cv2.imread("prac02ex05img01.png", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Cannot load 'prac02ex05img01.png'")

# -------- Apply histogram equalization --------
equalized_image = cv2.equalizeHist(image)

# -------- Compute histograms --------
hist_original  = cv2.calcHist([image], [0], None, [256], [0, 256])
hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

# -------- Compute cumulative distributions (CDFs) --------
cdf_original  = hist_original.cumsum()
cdf_equalized = hist_equalized.cumsum()

# Normalize CDFs for plotting (scale 0-1)
cdf_original_norm  = cdf_original / cdf_original.max()
cdf_equalized_norm = cdf_equalized / cdf_equalized.max()

# -------- Plot --------
plt.figure(figsize=(15, 10))

# Images
plt.subplot(3, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(3, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

# Histograms
plt.subplot(3, 2, 3)
plt.plot(hist_original, color='blue')
plt.title('Original Histogram')

plt.subplot(3, 2, 4)
plt.plot(hist_equalized, color='green')
plt.title('Equalized Histogram')

# Cumulative Distributions
plt.subplot(3, 2, 5)
plt.plot(cdf_original_norm, color='blue')
plt.title('Original Cumulative Distribution')

plt.subplot(3, 2, 6)
plt.plot(cdf_equalized_norm, color='green')
plt.title('Equalized Cumulative Distribution')

plt.tight_layout()
output_filename = "ex05_result.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_filename}")
