'''
Machine Perception Prac01 
Daehwan Yeo  

Exercise 2 - Read the image, extract the rectangular region.
'''

import os
import cv2 as cv
import numpy as np

# ---- Paths (same directory) ----
img_path = "./prac01ex02img01.png"
txt_path = "./prac01ex02crop.txt"
out_path = "./prac01ex02_result.png"

# ---- Load image ----
img = cv.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not read image at: {img_path}")

# ---- Read & parse coordinates ----
with open(txt_path, 'r') as file:
    first_line = file.readline().strip()

# Accept formats like: "xl yl xr yr" or "xl,yl,xr,yr"
tokens = first_line.replace(',', ' ').split()
if len(tokens) < 4:
    raise ValueError(f"Expected 4 numbers in {txt_path}, got: {first_line}")

xl, yl, xr, yr = map(int, tokens[:4])

# ---- Normalize and clamp coordinates ----
h, w = img.shape[:2]

# Ensure left<=right, top<=bottom
x1, x2 = sorted((xl, xr))
y1, y2 = sorted((yl, yr))

# Clamp to image bounds
x1 = max(0, min(x1, w - 1))
x2 = max(0, min(x2, w - 1))
y1 = max(0, min(y1, h - 1))
y2 = max(0, min(y2, h - 1))

# If coordinates are identical or invalid after clamping, bail out
if x2 <= x1 or y2 <= y1:
    raise ValueError(f"Invalid crop box after normalization/clamping: ({x1}, {y1}) to ({x2}, {y2})")

# NOTE: If the text file’s bottom-right is inclusive, use y2+1 and x2+1 below.
# Here we’ll treat xr,yr as inclusive corners and add +1 on slicing:
crop = img[y1:y2+1, x1:x2+1]

# ---- Save result ----
ok = cv.imwrite(out_path, crop)
if not ok:
    raise IOError(f"Failed to save cropped image to: {out_path}")

print(f"Cropped region saved to: {out_path}")
print(f"Crop box (inclusive): x=[{x1},{x2}], y=[{y1},{y2}]  -> shape: {crop.shape}")
