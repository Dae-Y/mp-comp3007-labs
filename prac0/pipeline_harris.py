import numpy as np
from skimage import io, color, filters
from skimage.feature import corner_harris, corner_peaks
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Input block: Read an image
class InputBlock:
    def __init__(self, image_path):
        self.image = io.imread(image_path)

    def get_image(self):
        return self.image

# Image processing block: grayscale + smoothing
class ImageProcessingBlock:
    def __init__(self, image):
        self.image = image

    def to_grayscale_and_smooth(self, sigma=1.0):
        img = self.image
        if img.ndim > 2:
            img = color.rgb2gray(img[..., :3])
        # light smoothing to reduce spurious corners
        return filters.gaussian(img, sigma=sigma, preserve_range=True)

# Feature extraction block: Harris + corner clustering
class FeatureExtractionBlock:
    def __init__(self, image):
        self.image = image

    def detect_corners(self, min_distance=5, threshold_rel=0.02, cluster_eps=8):
        # Harris corner response
        response = corner_harris(self.image)
        # raw detections (may include multiple points per true vertex)
        coords = corner_peaks(response, min_distance=min_distance, threshold_rel=threshold_rel)
        if coords.size == 0:
            return coords  # nothing found

        # cluster nearby detections so each geometric corner counts once
        # eps is in pixels; min_samples=1 so isolated points still form a cluster
        db = DBSCAN(eps=cluster_eps, min_samples=1).fit(coords)
        labels = db.labels_

        # cluster centers (mean of each cluster)
        clustered = np.vstack([coords[labels == k].mean(axis=0) for k in np.unique(labels)])
        return clustered

# Recognition block: simple rule on deduped corner count
class RecognitionBlock:
    def __init__(self, corner_points):
        self.corner_points = corner_points

    def recognize_shape(self):
        n = len(self.corner_points)
        # basic logic: 3 -> triangle, 4 -> rectangle, else circle (or 'other' if you like)
        if n == 3:
            return "Triangle"
        elif n == 4:
            return "Rectangle"
        else:
            return "Circle"

# --- Perception Pipeline Execution ---

image_path = '00_square.png'
#image_path = '00_triangle.png'
#image_path = '00_circle.png'

input_block = InputBlock(image_path)
image = input_block.get_image()

processing_block = ImageProcessingBlock(image)
gray_smooth = processing_block.to_grayscale_and_smooth(sigma=1.0)

extraction_block = FeatureExtractionBlock(gray_smooth)
corners = extraction_block.detect_corners(min_distance=5, threshold_rel=0.02, cluster_eps=8)

recognition_block = RecognitionBlock(corners)
shape = recognition_block.recognize_shape()
print("The recognized shape is:", shape)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.imshow(gray_smooth, cmap='gray')
if len(corners) > 0:
    ax.plot(corners[:, 1], corners[:, 0], 'r.', markersize=10)
ax.set_title(f'Recognized Shape: {shape} (corners: {len(corners)})')
ax.axis('off')

output_filename = 'harris_result.png'
plt.savefig(output_filename, bbox_inches='tight')
print(f"Result saved to {output_filename}")
