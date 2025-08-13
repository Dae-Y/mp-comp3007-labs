'''
Machine perception prac4
Daehwan Yeo

Exercise 1 - SIFT Feature Extraction
To gain hands-on experience with the SIFT algorithm, understanding its robustness 
against transformations and its ability to capture distinctive features in images. 
By comparing it with other methods like Harris corners, 
you'll appreciate the nuances and strengths of each approach.
'''
import numpy as np # Numpy library provides various useful functions and operators for scientific computing
import cv2 # openCV is a key library that provides various useful functions for computer vision
import os # Honestly this one is a bit optional.
import glob # again just optional
from matplotlib import pyplot as plt

# Load the image
filename = "prac04ex01img01.png"
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# --- Step 1 & 2: Keypoint Detection and Descriptor Extraction ---

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect SIFT keypoints and descriptors for the original image
# The detectAndCompute function finds keypoints and computes their descriptors.
keypoints, descriptors = sift.detectAndCompute(img, None)

print(f"Original Image: Found {len(keypoints)} keypoints.")
print(f"Original Descriptors Shape: {descriptors.shape}") # Should be (num_keypoints, 128)

# --- Step 3: Invariance Test with Rotated and Scaled Image ---

# Rotate and scale the image: 10-degree rotation and 1.2x scaling
rows, cols = img.shape
# Get the transformation matrix for rotation and scaling
# cv2.getRotationMatrix2D(center, angle, scale)
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1.2)
# Apply the affine transformation
# cv2.warpAffine(source_image, transformation_matrix, output_image_size)
rotated_scaled_img = cv2.warpAffine(img, M, (cols, rows))

# Detect SIFT keypoints and descriptors for the rotated and scaled image
rotated_keypoints, rotated_descriptors = sift.detectAndCompute(rotated_scaled_img, None)

print(f"\nTransformed Image: Found {len(rotated_keypoints)} keypoints.")
print(f"Transformed Descriptors Shape: {rotated_descriptors.shape}")

# --- Visualization ---

# Draw keypoints on the original and rotated/scaled images
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS draws keypoints with their size and orientation.
img_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
rotated_img_keypoints = cv2.drawKeypoints(rotated_scaled_img, rotated_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Plot the original and rotated/scaled images with keypoints side by side
plt.figure(figsize=(20, 10))
plt.subplot(121), plt.imshow(img_keypoints, 'gray'), plt.title('Original Image with SIFT Keypoints')
plt.subplot(122), plt.imshow(rotated_img_keypoints, 'gray'), plt.title('Rotated & Scaled Image with SIFT Keypoints')
plt.savefig('ex1_keypoints.png', bbox_inches='tight') # Save the figure
plt.close() # Close the plot to free up memory


# Plot the descriptors as intensity images side by side
# This helps visualize the patterns within the descriptor vectors.
plt.figure(figsize=(20, 10))
plt.subplot(121), plt.imshow(descriptors, 'gray'), plt.title('Original Image Descriptors')
plt.subplot(122), plt.imshow(rotated_descriptors, 'gray'), plt.title('Transformed Image Descriptors')
plt.savefig('ex1_descriptors.png', bbox_inches='tight') # Save the figure
plt.close()

# --- Extra: Perform keypoint matching ---

# Initialize the Brute-Force Matcher with default params
# cv2.NORM_L2 is used for SIFT/SURF. crossCheck=True returns only consistent pairs.
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors between the two images
matches = bf.match(descriptors, rotated_descriptors)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw the top 10 matches between the two images
# cv2.drawMatches links the corresponding keypoints in the two images.
img_matches = cv2.drawMatches(img, keypoints, rotated_scaled_img, rotated_keypoints, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matched keypoints
plt.figure(figsize=(20, 10))
plt.imshow(img_matches)
plt.title('Top 10 SIFT Feature Matches')
plt.savefig('ex1_matches.png', bbox_inches='tight') # Save the figure
plt.close()

print("\nSuccessfully saved output images as ex1_keypoints.png, ex1_descriptors.png, and ex1_matches.png")