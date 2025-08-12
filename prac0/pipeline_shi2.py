'''
Machine perception prac0 
Daehwan Yeo

A simple perception pipeline using object-oriented programming.
The pipeline will include the following blocks:
- Input block: Read an image that consists of a rectangle or circle.
- Image processing block: Perform filtering (image smoothing).
- Feature extraction block: Perform corner detection.
- Recognition block: Perform simple recognition using decision trees based on the number of corners.

Slightly updated version for detecting triangles using the Shi-Tomasi method
Not working
'''
import numpy as np
from skimage import io, filters, feature, color, transform
from skimage.feature import corner_harris, corner_subpix
from skimage.feature import corner_shi_tomasi, corner_peaks
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Input block: Read an image
class InputBlock:
    def __init__(self, image_path):
        # Loads an image from the specified path.
        self.image = io.imread(image_path)

    def get_image(self):
        return self.image


# Image processing block: Convert to grayscale
class ImageProcessingBlock:
    def __init__(self, image):
        # Stores the image to be processed.
        self.image = image

    def convert_to_grayscale(self):
        # Converts the stored image to grayscale if it's not already.
        # Check if the image has more than 2 dimensions (i.e., is color)
        if self.image.ndim > 2:
            # Convert to grayscale, handling potential alpha channel
            return color.rgb2gray(self.image[..., :3])
        return self.image
        
        
# Feature extraction block: Perform corner detection
class FeatureExtractionBlock:
    def __init__(self, image, method='shi', use_smoothing=True):
        self.image = image
        self.method = method
        self.use_smoothing = use_smoothing

    def detect_corners(self):
        # Optional smoothing to reduce noise
        if self.use_smoothing:
            smoothed = filters.gaussian(self.image, sigma=1)
        else:
            smoothed = self.image

        if self.method == 'shi':
            response = corner_shi_tomasi(smoothed)
        elif self.method == 'harris':
            response = corner_harris(smoothed)
        else:
            raise ValueError("Unsupported method. Use 'shi' or 'harris'.")

        # Tune corner_peaks for better triangle detection
        coords = corner_peaks(response, min_distance=5, threshold_rel=0.01)
        return coords

        
        
# Recognition block: Perform simple recognition
class RecognitionBlock:
    def __init__(self, coords):
        # Stores the corner coordinates for recognition.
        self.coords = coords

    def recognize_shape(self):
        # Recognizes the shape based on the number of corners.
        num_corners = len(self.coords)
        
        # This is a simple decision tree logic
        if 2 <= num_corners <= 3:
            return "Triangle"
        elif 4 <= num_corners <= 5:
            return "Rectangle"
        elif num_corners > 5:
            return "Circle"
        else:
            return "Unknown"


# --- Perception Pipeline Execution ---

# Create and test the perception pipeline
#image_path = '00_square.png'  # square class
image_path = '00_triangle.png' # triangle class
#image_path = '00_circle.png' # circle class

# Input block
input_block = InputBlock(image_path)
image = input_block.get_image()

# Image processing block
processing_block = ImageProcessingBlock(image)
grayscale_image = processing_block.convert_to_grayscale()

# Feature extraction block
extraction_block = FeatureExtractionBlock(grayscale_image)
corners = extraction_block.detect_corners()

# Recognition block
recognition_block = RecognitionBlock(corners)
shape = recognition_block.recognize_shape()
print("The recognized shape is:", shape)

# Plot the results
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.imshow(grayscale_image, cmap='gray')
ax.plot(corners[:, 1], corners[:, 0], 'r.')
ax.set_title(f'Recognized Shape: {shape}')
ax.axis('off')

# Save the plot to a .png file
output_filename = 'shi2_result.png'
plt.savefig(output_filename)
print(f"Result saved to {output_filename}")