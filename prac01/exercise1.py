'''
Machine Perception Prac01 
Daehwan Yeo  

Exercise 1 - Basic Input/Output, Histogram, and Resizing  
'''

# Write a program that reads all colour images in a given directory, one by one.
# For each image
#  - Print out the image file name and its dimensions (width and height).
#  - Compute and print out the histogram (with 10 uniform bins) of each colour channel (R, G, B).
#    Make an observation of the histograms.
#  - Reduce the size of the input image by 50% and output this as an image file of the same format.
#  - Test the program with the images provided (i.e. prac01ex01imgXX.png).

import os
import cv2 as cv
import numpy as np

# Set up paths
path = 'ex01'
output_folder = path
os.makedirs(output_folder, exist_ok=True)

# Define target image filenames
filenames = [f"prac01ex01img0{i}.png" for i in range(1, 6)]

# Loop through each image
for filename in filenames:
    filepath = os.path.join(path, filename)
    print(f"\nProcessing file: {filename}")

    img = cv.imread(filepath)
    if img is None:
        print("  (Could not read image; skipping.)")
        continue


    # --- PART 1: Image Info ---
    
    # compute height, width and channels
    height, width = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    print("  Dimensions → Height:", height, "Width:", width, "Channels:", channels)


    # --- PART 2: Histogram (10 bins per channel) ---
    # References: https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html

    # split the image into 3 channels, note the BGR order in OpenCV
    B, G, R = cv.split(img)
    
    # compute the histogram for each channel (10 uniform bins over [0, 256))
    hist_B = cv.calcHist([B], [0], None, [10], [0, 256])
    hist_G = cv.calcHist([G], [0], None, [10], [0, 256])
    hist_R = cv.calcHist([R], [0], None, [10], [0, 256])

    print("  Histogram (Blue):", np.transpose(hist_B))
    print("  Histogram (Green):", np.transpose(hist_G))
    print("  Histogram (Red):", np.transpose(hist_R))


    # --- PART 3: Resize and Save ---
    # References: https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
    
    # image resize (reduce size by 50%)
    img_resized = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    
    # save the image to ex01 folder with suffix "_half"
    filename_out = os.path.join(output_folder, filename.replace(".png", "_half.png"))
    success = cv.imwrite(filename_out, img_resized)

    if success:
        print("  Saved resized image to:", filename_out)
    else:
        print("  Failed to save resized image.")
