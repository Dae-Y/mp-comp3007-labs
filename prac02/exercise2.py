'''
Machine Perception Prac02
Daehwan Yeo  

Exercise 2 - Linear Filtering

Use the filter2D function in OpenCV to perform linear filtering 
on a gray-scale image with various kernels / filters.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------- Load grayscale image (same file as Exercise 1) --------
img = cv2.imread("prac02ex01img01.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Cannot load 'prac02ex01img01.jpg'")

# -------- Define kernels --------
# Prewitt (3x3)
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)

prewitt_y = np.array([[-1, -1, -1],
                      [ 0,  0,  0],
                      [ 1,  1,  1]], dtype=np.float32)

# Sobel (3x3)
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

# Laplacian (4-neighbour)
laplacian = np.array([[0,  1, 0],
                      [1, -4, 1],
                      [0,  1, 0]], dtype=np.float32)

# Gaussian σ=1 (5x5) via separable vector -> 2D kernel
g1d = cv2.getGaussianKernel(ksize=5, sigma=1)  # column vector
gaussian = (g1d @ g1d.T).astype(np.float32)

# -------- Apply filters (use CV_64F for derivatives, then abs & convert) --------
img_prewitt_x_64 = cv2.filter2D(img, ddepth=cv2.CV_64F, kernel=prewitt_x)
img_prewitt_y_64 = cv2.filter2D(img, ddepth=cv2.CV_64F, kernel=prewitt_y)
img_prewit_x = cv2.convertScaleAbs(img_prewitt_x_64)
img_prewit_y = cv2.convertScaleAbs(img_prewitt_y_64)

img_sobel_x_64 = cv2.filter2D(img, ddepth=cv2.CV_64F, kernel=sobel_x)
img_sobel_y_64 = cv2.filter2D(img, ddepth=cv2.CV_64F, kernel=sobel_y)
img_sobel_x = cv2.convertScaleAbs(img_sobel_x_64)
img_sobel_y = cv2.convertScaleAbs(img_sobel_y_64)

img_laplacian_64 = cv2.filter2D(img, ddepth=cv2.CV_64F, kernel=laplacian)
img_laplacian = cv2.convertScaleAbs(img_laplacian_64)

# Gaussian smoothing (no negatives → keep uint8)
img_gaussian = cv2.filter2D(img, ddepth=-1, kernel=gaussian)

# -------- Compare with GaussianBlur at different σ --------
img_gaussian_blur_small_sigma = cv2.GaussianBlur(img, (5, 5), sigmaX=1)   # similar to 'gaussian' above
img_gaussian_blur_large_sigma = cv2.GaussianBlur(img, (11, 11), sigmaX=3) # stronger blur

# -------- Plot --------
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

axes[0, 0].imshow(img, cmap='gray');                axes[0, 0].set_title('Original')
axes[0, 1].imshow(img_prewit_x, cmap='gray');       axes[0, 1].set_title('Prewitt X')
axes[0, 2].imshow(img_prewit_y, cmap='gray');       axes[0, 2].set_title('Prewitt Y')

axes[1, 0].imshow(img_sobel_x, cmap='gray');        axes[1, 0].set_title('Sobel X')
axes[1, 1].imshow(img_sobel_y, cmap='gray');        axes[1, 1].set_title('Sobel Y')
axes[1, 2].imshow(img_laplacian, cmap='gray');      axes[1, 2].set_title('Laplacian')

axes[2, 0].imshow(img_gaussian, cmap='gray');       axes[2, 0].set_title('Gaussian (σ=1, 5x5 via filter2D)')
axes[2, 1].imshow(img_gaussian_blur_small_sigma, cmap='gray'); axes[2, 1].set_title('GaussianBlur σ=1 (5x5)')
axes[2, 2].imshow(img_gaussian_blur_large_sigma, cmap='gray'); axes[2, 2].set_title('GaussianBlur σ=3 (11x11)')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
output_filename = "ex02_result.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_filename}")