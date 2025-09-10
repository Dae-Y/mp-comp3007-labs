"""
Machine Perception Prac02
Daehwan Yeo

Exercise 4 - Pixel Transform

Performs an affine transformation on all pixels of a gray-scale image 
that suffers from poor contrast and brightness.
The goal is to enhance the image by adjusting its contrast and brightness.
Enhance brightness and contrast using affine transformation:
g(x,y) = α * f(x,y) + β
"""

import cv2
import matplotlib.pyplot as plt

# -------- Load image --------
img_gray = cv2.imread("prac02ex04img01.png", cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    raise FileNotFoundError("Cannot load 'prac02ex04img01.png'")

# -------- Parameters for affine transform --------
alpha = 2.0  # Contrast control (>1 = higher contrast, 0-1 = lower contrast)
beta  = 50   # Brightness control (positive = brighter, negative = darker)

# -------- Apply affine transform --------
img_affine = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=beta)

# -------- Plot --------
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_affine, cmap='gray', vmin=0, vmax=255)
plt.title(f'Enhanced (α={alpha}, β={beta})')
plt.axis('off')

plt.tight_layout()
output_filename = "ex04_result.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_filename}")
