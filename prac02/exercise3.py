"""
Machine Perception Prac02
Daehwan Yeo

Exercise 3 - Median Filtering

Applies median filtering to remove salt-and-pepper noise.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------- Load images --------
img_org  = cv2.imread("prac02ex02img01.jpg", cv2.IMREAD_GRAYSCALE)  # clean reference image
img_gray = cv2.imread("prac02ex02img02.jpg", cv2.IMREAD_GRAYSCALE)  # corrupted with noise

if img_org is None or img_gray is None:
    raise FileNotFoundError("Cannot load input images.")

# -------- Apply Median Filtering with different kernel sizes --------
img_filtered_k3 = cv2.medianBlur(img_gray, 3)
img_filtered_k5 = cv2.medianBlur(img_gray, 5)

# -------- Plot --------
fig, axes = plt.subplots(1, 4, figsize=(16, 8))

axes[0].imshow(img_org, cmap='gray')
axes[0].set_title("Original img")
axes[0].axis('off')

axes[1].imshow(img_gray, cmap='gray')
axes[1].set_title("Noisy img")
axes[1].axis('off')

axes[2].imshow(img_filtered_k3, cmap='gray')
axes[2].set_title("Median filtered k=3")
axes[2].axis('off')

axes[3].imshow(img_filtered_k5, cmap='gray')
axes[3].set_title("Median filtered k=5")
axes[3].axis('off')

plt.tight_layout()
output_filename = "ex03_result.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_filename}")
