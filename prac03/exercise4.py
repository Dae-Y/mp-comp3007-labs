"""
Machine Perception Prac03
Daehwan Yeo

Exercise 4 - Blob Detection (MSER)

Tuning tips:

_delta: smaller → more thresholds swept → more regions (good for faint blobs, might add noise).

min_area / max_area: prune tiny specks or huge swaths (e.g., background). 
  Start with min_area≈50-150, max_area_ratio≈0.2-0.4.

Coins: uniform blobs → lower min_area, modest delta.

Playing card: card face features will form multiple MSERs
  raise min_area to avoid tiny text/dots if not desired.

Card on carpet: textures can create many small MSERs
  raise min_area and maybe delta.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Images to test (coins, single card, card on carpet)
images_path = [
    "prac03ex04img01.png",
    "prac03ex04img02.png",
    "prac03ex04img03.png",
]

def mser_image(image_path, delta=5, min_area=60, max_area_ratio=0.25):
    """
    Detects MSER regions and returns an RGB image with convex hulls overlaid.
    - delta: intensity step between thresholds (smaller -> more regions)
    - min_area: minimum region size in pixels (filter tiny noise)
    - max_area_ratio: max region size as a fraction of image area
    """
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load '{image_path}'")

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Compute absolute max area from ratio
    H, W = img_gray.shape
    max_area = int(max_area_ratio * H * W)

    # MSER detector (use setters for portability across OpenCV versions)
    mser = cv2.MSER_create()
    mser.setDelta(delta)
    mser.setMinArea(int(min_area))
    mser.setMaxArea(int(max_area))

    # Detect regions (regions: list of Nx2 point arrays; boxes not used here)
    regions, _ = mser.detectRegions(img_gray)

    # Draw convex hulls for each region
    overlay = img_bgr.copy()
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions if len(p) >= 3]
    cv2.polylines(overlay, hulls, isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    # Optional: draw a fitted ellipse for larger regions (nice visualization)
    # for pts in regions:
    #     if len(pts) >= 5:  # fitEllipse needs at least 5 points
    #         ellipse = cv2.fitEllipse(pts)
    #         cv2.ellipse(overlay, ellipse, (255, 0, 0), 2, cv2.LINE_AA)

    # Convert to RGB for matplotlib
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# -------- Plot all results --------
plt.figure(figsize=(18, 6))
for i, path in enumerate(images_path, start=1):
    try:
        # Tune per image if needed by passing delta/min_area/max_area_ratio here
        vis = mser_image(path, delta=5, min_area=80, max_area_ratio=0.3)
        plt.subplot(1, 3, i)
        plt.imshow(vis)
        plt.title(f"Detected MSER regions\n{path}")
        plt.axis('off')
    except FileNotFoundError as e:
        plt.subplot(1, 3, i)
        plt.text(0.5, 0.5, str(e), ha='center', va='center', fontsize=10)
        plt.axis('off')
        plt.title("Missing image")

plt.tight_layout()
plt.savefig("ex04_result.png", dpi=300, bbox_inches='tight')
print("Saved: ex04_result.png")
plt.close()
