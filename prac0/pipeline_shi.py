'''
Machine perception prac0 
Daehwan Yeo

A simple perception pipeline using object-oriented programming.
The pipeline will include the following blocks:
- Input block: Read an image that consists of a rectangle or circle.
- Image processing block: Perform filtering (image smoothing).
- Feature extraction block: Perform corner detection.
- Recognition block: Perform simple recognition using decision trees based on the number of corners.
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
    def __init__(self, image):
        # Stores the grayscale image for feature extraction.
        self.image = image

    def detect_corners(self):
        # Detects corners using the Shi-Tomasi method.
        # Use Shi-Tomasi corner detector, which is good for tracking features
        # corner_peaks finds coordinates of corners from the corner response image
        coords = feature.corner_peaks(feature.corner_shi_tomasi(self.image), min_distance=10)
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
        if num_corners == 3:
            return "Triangle"
        elif num_corners == 4:
            return "Rectangle"
        else:
            # Assumes shapes with very few (<3) or many detected corners are circles
            return "Circle"


# --- Perception Pipeline Execution ---

# Create and test the perception pipeline
#image_path = '00_square.png'  # square class
#image_path = '00_triangle.png' # triangle class
image_path = '00_circle.png' # circle class

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
output_filename = 'shi_result.png'
plt.savefig(output_filename)
print(f"Result saved to {output_filename}")