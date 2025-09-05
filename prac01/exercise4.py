'''
Machine Perception Prac01
Daehwan Yeo

Exercise 4 - Rotate an image back to original orientation and find the optimal angle.
'''

import math
import cv2 as cv
import numpy as np

# References: https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html

# --- Load image (same folder) ---
path = "./prac01ex04img01.png"
img = cv.imread(path)
if img is None:
    raise FileNotFoundError(f"Could not read image at: {path}")


# complete a function to rotate and show an image
def rotate_image(img, theta):
    # image dimensions
    h, w = img.shape[:2]

    # center of rotation
    center = (w // 2, h // 2)

    # create the rotation matrix (Positive theta = CCW)
    M = cv.getRotationMatrix2D(center, theta, 1.0)

    # perform the rotation (keep same canvas size; black border)
    rotated = cv.warpAffine(
        img, M, (w, h),
        flags=cv.INTER_CUBIC,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # The function should return the rotated image
    return rotated


# manually rotate the image
theta = 10
rot_demo = rotate_image(img, theta)
cv.imwrite("./prac01ex04_manual_10deg.png", rot_demo)
print("Saved: ./prac01ex04_manual_10deg.png")


# Challenge: to find optimal angle, just compute the "bounding box"
# of the original rotated image. Make use of black pixels

# convert to gray scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# create a binary mask of non-black pixels (tolerant threshold)
# (black corners from prior rotation are near 0)
_, mask = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)

# find contours from the binary mask
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# assume the largest contour corresponds to the object
cnt = max(contours, key=cv.contourArea)

# fit a minimum area rectangle to the contour
rect = cv.minAreaRect(cnt)

# extract the angle of the rectangle
theta_optimal = rect[-1]

# OpenCV returns angles in range [-90, 0)
# Adjust to get a proper rotation angle
if theta_optimal < -45:
    theta_optimal = 90 + theta_optimal

print("Optimal angle found:", theta_optimal)

# rotate image with optimal angle and save
img_corrected = rotate_image(img, theta_optimal)

cv.imwrite("./prac01ex04_optimal_result.png", img_corrected)
print("Saved: ./prac01ex04_optimal_result.png")
