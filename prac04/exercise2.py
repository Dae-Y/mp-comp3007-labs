'''
Machine perception prac4
Daehwan Yeo

Exercise 2 - Binary Shape Analysis
Working with binary shape analysis to extract 
blob features from a gray-scale input image. 
The main goal is to separate individual characters 
from the image and then extract several binary features from them.
'''
import numpy as np # Numpy library provides various useful functions and operators for scientific computing
import cv2 # openCV is a key library that provides various useful functions for computer vision
import os # Honestly this one is a bit optional.
import glob # again just optional
from matplotlib import pyplot as plt

# Complete the following functions for the Exercise

def display_image(img, title="", save_path=None):
    """
    Display an image with matplotlib.
    If save_path is provided, save to that PNG file instead of calling plt.show().
    """
    plt.figure()
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        # Convert BGR→RGB for correct display
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

    if save_path:
        # save to disk
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved '{title}' as '{save_path}'")
    else:
        plt.show()
    plt.close()

def imshow_components(labels):
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0
    return labeled_img

def extract_features(labels_im, stats):
    features = []
    num_labels = stats.shape[0]
    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        bbox = (left, top, width, height)
        ys, xs = np.where(labels_im == i)
        centroid = (np.mean(xs), np.mean(ys))
        features.append({
            "Label": i,
            "Area": area,
            "BoundingBox": bbox,
            "Centroid": centroid
        })
    return features

# ──────────────────────────────────────────────────────────────────────────────
# Main script

# 1) Load & show original
filename = "prac04ex02img01.png"
img = cv2.imread(filename, cv2.IMREAD_COLOR)
display_image(img, "Original Image", save_path="ex2_original.png")  # ← save original

# 2) Convert + blur
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 3) Threshold variants
_, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, th2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

display_image(th1, "Basic Binary Thresholding", save_path="ex2_basic_threshold.png")  # ← save basic
display_image(th2, "Otsu's Thresholding", save_path="ex2_otsu_threshold.png")        # ← save otsu
display_image(th3, "Gaussian Blur + Otsu's Thresholding", save_path="ex2_gaussian_otsu.png")  # ← save gaussian+otsu

# 4) Invert for CC analysis
th = cv2.bitwise_not(th3)

# 5) Connected components
num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)

# 6) Colorize and save result
colored_components_img = imshow_components(labels_im)
display_image(colored_components_img, "Colored Components", save_path="ex2_result.png")

# 7) Feature extraction
blob_features = extract_features(labels_im, stats)

# 8) Compare two blobs
blob1 = blob_features[0]
blob2 = blob_features[1]

print("Features for Blob 1:")
for k, v in blob1.items():
    print(f"  {k}: {v}")

print("\nFeatures for Blob 2:")
for k, v in blob2.items():
    print(f"  {k}: {v}")

diff_area = blob1["Area"] - blob2["Area"]
print(f"\nDifference in Area between Blob 1 and Blob 2: {diff_area}")
