'''
Machine perception prac0 
Daehwan Yeo
'''

print("--------- Exercise 1 Simple function ----------")
# Function to add two numbers
def add_numbers(a, b):
  """
  This function takes two numbers, a and b, as input 
  and returns their sum.
  """
  # Add your implementation below
  return a + b

# Test the function
result = add_numbers(3, 5)
print("The sum of 3 and 5 is:", result)



print("\n--------- Exercise 2 OOP ----------")
# Class to represent a point in 2D space
class Point:
  def __init__(self, x, y):
    """
    Initializes a point with coordinates x and y.
    """
    # Add your implementation below
    self.x = x
    self.y = y

  def move(self, dx, dy):
    """
    Moves the point by dx along the x-axis and dy along the y-axis.
    """
    # Add your implementation below
    self.x += dx
    self.y += dy

  def __str__(self):
    """
    Returns a string representation of the point.
    """
    return f"Point({self.x}, {self.y})"

# Test the class
p = Point(2, 3)
print("Initial position:", p)
p.move(1, -1)
print("Position after move:", p)



print("\n--------- Exercise 3 Plotting with Matplotlib ----------")
import numpy as np
import matplotlib.pyplot as plt

# Generate data
# Create an array of 200 points from 0 to 2*pi
x = np.linspace(0, 2 * np.pi, 200) 
y_sin = np.sin(x)  # Calculate the sine of x
y_cos = np.cos(x)  # Calculate the cosine of x

# Create the plot
fig = plt.figure(figsize=(10, 6)) # Create and assign the figure
plt.plot(x, y_sin, label='Sine Wave', color='blue') # Plot the sine wave
plt.plot(x, y_cos, label='Cosine Wave', color='red') # Plot the cosine wave

# Add plot details
plt.title('Sine and Cosine Waves')
plt.xlabel('x-axis (radians)')
plt.ylabel('y-axis')
plt.grid(True)
plt.legend()

# Display the plot
# plt.show()
# Instead of plt.show(), save the figure to a file
output_filename = "sine_cosine_waves.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight') # dpi for resolution, bbox_inches for tight layout
print(f"Plot saved to {output_filename}")
plt.close(fig) # Close the figure to free up memory



print("\n--------- Exercise 4 Image Processing with Pillow ----------")
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

try:
  # Load the image and convert to grayscale
  original_image = Image.open('fig_superlab.jpg').convert('L')

  # Perform histogram equalization
  # This enhances the contrast of the image
  equalized_image = ImageOps.equalize(original_image)

  # Plot the original and equalized images
  # Create a figure to hold the subplots
  fig = plt.figure(figsize=(12, 6)) # Assign the figure to a variable

  # Subplot 1: Original Image
  plt.subplot(1, 2, 1)
  plt.imshow(original_image, cmap='gray')
  plt.title('Original Image')
  plt.axis('off') # Hide the axes

  # Subplot 2: Equalized Image
  plt.subplot(1, 2, 2)
  plt.imshow(equalized_image, cmap='gray')
  plt.title('Histogram Equalized Image')
  plt.axis('off') # Hide the axes
  
  # Adjust layout and save the figure
  plt.tight_layout()
  output_filename = "equalised_image.png"
  plt.savefig(output_filename, dpi=300, bbox_inches='tight')
  print(f"Equalised image saved to {output_filename}")
  plt.close(fig)

except FileNotFoundError:
  print("Error: 'fig_superlab.jpg' not found. Please ensure the image is in the correct directory.")