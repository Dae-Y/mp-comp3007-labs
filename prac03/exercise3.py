"""
Machine Perception Prac03
Daehwan Yeo

Exercise 3 - Line Detection
Hough Transform

Threshold in HoughLines:
- It's the minimum number of votes a bin in the accumulator 
  must have to be considered a valid line.
- Higher threshold → fewer, stronger lines detected.
- Lower threshold → more lines (including weak/noisy ones).

Why edges first?
- Hough assumes sparse edge points. Without edge detection (e.g., using raw grayscale), 
  the accumulator is flooded → false positives and massive computation.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Part 1: Custom Hough Transform Accumulator
# ------------------------
def custom_hough_transform(edge_img, theta_res=1, rho_res=1):
    """
    Compute Hough accumulator for a binary edge image.
    """
    h, w = edge_img.shape
    max_dist = int(np.hypot(h, w))  # diagonal distance
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    rhos = np.arange(-max_dist, max_dist + 1, rho_res)

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    ys, xs = np.nonzero(edge_img)  # edge points
    for (x, y) in zip(xs, ys):
        for t_idx, theta in enumerate(thetas):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            r_idx = rho + max_dist
            accumulator[r_idx, t_idx] += 1

    return accumulator, thetas, rhos

# ------------------------
# Part 2: Line Detection with OpenCV
# ------------------------
# Choose image
# image_path = "prac03ex01img01.png"   # Checkerboard
# image_path = "prac03ex03img01.png"   # Diamond
image_path = "prac03ex03img02.jpg"     # Natural building image

RGB = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
RGB_original = RGB.copy()

if RGB is None or gray is None:
    raise FileNotFoundError(f"Could not load {image_path}")

# Step 1: Edge detection
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

# Step 2: Custom Hough accumulator (for visualization)
accumulator, thetas, rhos = custom_hough_transform(edges)

plt.figure(figsize=(12, 5))
plt.imshow(accumulator, cmap='hot', aspect='auto',
           extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]),
                   rhos[-1], rhos[0]])
plt.title("Hough Accumulator")
plt.xlabel("Theta (degrees)")
plt.ylabel("Rho (pixels)")
plt.colorbar()
plt.savefig("ex03_accumulator.png", dpi=300, bbox_inches='tight')
print("Saved: ex03_accumulator.png")
plt.close()

# Step 3: Use OpenCV's HoughLines (standard)
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=150)

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(RGB, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Step 4: Probabilistic Hough (HoughLinesP)
linesP = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                         threshold=80, minLineLength=50, maxLineGap=10)

if linesP is not None:
    for line in linesP:
        x1, y1, x2, y2 = line[0]
        cv2.line(RGB, (x1, y1), (x2, y2), (255, 0, 0), 2)

# ------------------------
# Display final results
# ------------------------
plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(RGB_original, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edges")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(RGB, cv2.COLOR_BGR2RGB))
plt.title("Detected Lines (Green=Hough, Blue=Probabilistic)")
plt.axis("off")

plt.tight_layout()
plt.savefig("ex03_result.png", dpi=300, bbox_inches='tight')
print("Saved: ex03_result.png")
plt.close()
