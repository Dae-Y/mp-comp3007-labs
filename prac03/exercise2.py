"""
Machine Perception Prac03
Daehwan Yeo

Exercise 2 - Edge Detection
Canny edge detection method

Gradient magnitude:
- Sobel: stronger edge responses, smoother.
- Prewitt: simpler, weaker but sharper responses.

Canny thresholds:
- cv2.Canny(img, 50, 150) → detects more edges (more noise).
- cv2.Canny(img, 150, 250) → fewer, cleaner edges.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------- Load image --------
# Checkerboard (synthetic edges)
# img = cv2.imread("prac03ex01img01.png", cv2.IMREAD_GRAYSCALE)
# Natural image (vehicle)
img = cv2.imread("prac03ex02img01.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image could not be loaded.")

# -------- Define Prewitt kernels --------
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)

prewitt_y = np.array([[-1, -1, -1],
                      [ 0,  0,  0],
                      [ 1,  1,  1]], dtype=np.float32)

# -------- Define Sobel kernels --------
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

# -------- Convolution with Sobel and Prewitt --------
sobel_gx = cv2.filter2D(img, cv2.CV_64F, sobel_x)
sobel_gy = cv2.filter2D(img, cv2.CV_64F, sobel_y)

prewitt_gx = cv2.filter2D(img, cv2.CV_64F, prewitt_x)
prewitt_gy = cv2.filter2D(img, cv2.CV_64F, prewitt_y)

# -------- Gradient Magnitude --------
sobel_magnitude = cv2.magnitude(sobel_gx, sobel_gy)
prewitt_magnitude = cv2.magnitude(prewitt_gx, prewitt_gy)

# Normalize to 8-bit for display
sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)
prewitt_magnitude = cv2.convertScaleAbs(prewitt_magnitude)

# -------- Display gradient results --------
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sobel_magnitude, cmap='gray')
plt.title("Sobel Gradient Magnitude")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(prewitt_magnitude, cmap='gray')
plt.title("Prewitt Gradient Magnitude")
plt.axis('off')

plt.tight_layout()
plt.savefig("ex02_gradients.png", dpi=300, bbox_inches='tight')
print("Saved: ex02_gradients.png")
plt.close()

# -------- Part 2: Canny Edge Detection --------
# Experiment with thresholds (low, high)
edges = cv2.Canny(img, threshold1=100, threshold2=200)

plt.figure(figsize=(6, 6))
plt.imshow(edges, cmap='gray')
plt.title("Canny Edges (100, 200)")
plt.axis('off')
plt.tight_layout()
plt.savefig("ex02_canny.png", dpi=300, bbox_inches='tight')
print("Saved: ex02_canny.png")
plt.close()
