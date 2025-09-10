"""
Machine Perception Prac02
Daehwan Yeo

Exercise 6 - Morphology

Morphological operations are fundamental tools in image processing, 
particularly in the areas of noise reduction, image enhancement, and object recognition.

Explore basic and advanced morphological operations using OpenCV.
"""
import cv2
import matplotlib.pyplot as plt

# -------- Load grayscale image --------
image = cv2.imread("prac02ex06img01.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Cannot load 'prac02ex06img01.png'")

# Invert if needed (white foreground on black background)
image = 255 - image

# -------- Define structuring element --------
kernel_size = 5
element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

# -------- Basic morphology --------
dilated = cv2.dilate(image, element)
eroded  = cv2.erode(image, element)

# -------- Advanced morphology --------
opened   = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
closed   = cv2.morphologyEx(image, cv2.MORPH_CLOSE, element)
gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, element)
blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, element)

# -------- Plot results --------
plt.figure(figsize=(20, 15))

plt.subplot(3, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(dilated, cmap='gray')
plt.title('Dilated')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(eroded, cmap='gray')
plt.title('Eroded')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(opened, cmap='gray')
plt.title('Opening')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(closed, cmap='gray')
plt.title('Closing')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(gradient, cmap='gray')
plt.title('Morphological Gradient')
plt.axis('off')

plt.subplot(3, 3, 7)
plt.imshow(blackhat, cmap='gray')
plt.title('Blackhat')
plt.axis('off')

plt.tight_layout()
output_filename = "ex06_result.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_filename}")
