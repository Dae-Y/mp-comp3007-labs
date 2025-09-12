"""
Machine perception prac4
Daehwan Yeo

Exercise 3 - Histogram Feature Extraction
Simplified: Use 256 bins, and save 2 images per character:
1) Histogram
2) Bounding box on original image
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Helper functions
# ---------------------------
def compute_histogram(image, bins=256):
    """Compute grayscale histogram of image patch (flattened)."""
    hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist = hist.flatten()
    # normalize
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist

def display_histogram(hist, title="Histogram", save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(hist, color='blue')
    plt.title(title)
    plt.xlabel("Bin (0-255)")
    plt.ylabel("Normalized Frequency")
    plt.grid(alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved {save_path}")
    else:
        plt.show()
    plt.close()

def display_image(img, title="", save_path=None):
    plt.figure()
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved {save_path}")
    else:
        plt.show()
    plt.close()

# ---------------------------
# Bootstrapping from Exercise 2
# ---------------------------
filename = "prac04ex02img01.png"

if not os.path.exists(filename):
    raise FileNotFoundError(f"Cannot find {filename} in {os.getcwd()}")

img = cv2.imread(filename, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Otsu + invert
_, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Connected components
num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)

# ---------------------------
# Extract bounding boxes and compute histograms
# ---------------------------
outdir = "ex3_outputs"
os.makedirs(outdir, exist_ok=True)

for k in range(1, num_labels):  # skip background
    x, y, w, h, area = stats[k]

    # crop character patch
    character_patch = gray[y:y+h, x:x+w]

    # compute histogram (256 bins)
    hist = compute_histogram(character_patch, bins=256)

    # save histogram plot
    display_histogram(
        hist,
        title=f"Histogram for Character {k}",
        save_path=os.path.join(outdir, f"hist_char{k}.png")
    )

    # draw bounding box on copy of original
    output = img.copy()
    cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # save image with bounding box
    display_image(
        output,
        title=f"Character {k} with Bounding Box",
        save_path=os.path.join(outdir, f"bbox_char{k}.png")
    )

print(f"Saved results in folder: {outdir}")
