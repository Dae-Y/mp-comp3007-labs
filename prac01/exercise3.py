'''
Machine Perception Prac01 
Daehwan Yeo  

Exercise 3 - Draw a bounding box over the specified region and mark corners.
'''

import os
import cv2 as cv

# --- Paths (same directory as files) ---
img_path = "./prac01ex02img01.png"
txt_path = "./prac01ex02crop.txt"
out_path = "./prac01ex03_result.png"

# --- Load image ---
img = cv.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not read image at: {img_path}")

# --- Read & parse coordinates ---
with open(txt_path, 'r') as file:
    first_line = file.readline().strip()

# Robust parsing: allow spaces or commas
tokens = first_line.replace(',', ' ').split()
if len(tokens) < 4:
    raise ValueError(f"Expected 4 numbers in {txt_path}, got: {first_line}")

xl, yl, xr, yr = map(int, tokens[:4])

# --- Normalize & clamp to image bounds ---
h, w = img.shape[:2]
x1, x2 = sorted((xl, xr))
y1, y2 = sorted((yl, yr))
x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))

if x2 <= x1 or y2 <= y1:
    raise ValueError(f"Invalid box after normalization: ({x1},{y1})-({x2},{y2})")

# --- Draw bounding box ---
box_color = (0, 255, 0)      # green
thickness = 2
cv.rectangle(img, (x1, y1), (x2, y2), box_color, thickness, lineType=cv.LINE_AA)

# --- Draw corner markers (filled circles) ---
marker_color = (0, 0, 255)   # red
radius = max(3, int(0.006 * max(w, h)))  # scale with image size
cv.circle(img, (x1, y1), radius, marker_color, -1, lineType=cv.LINE_AA)  # top-left
cv.circle(img, (x2, y1), radius, marker_color, -1, lineType=cv.LINE_AA)  # top-right
cv.circle(img, (x1, y2), radius, marker_color, -1, lineType=cv.LINE_AA)  # bottom-left
cv.circle(img, (x2, y2), radius, marker_color, -1, lineType=cv.LINE_AA)  # bottom-right

# --- Save output ---
ok = cv.imwrite(out_path, img)
if not ok:
    raise IOError(f"Failed to save output image to: {out_path}")

print(f"Saved: {out_path}")
