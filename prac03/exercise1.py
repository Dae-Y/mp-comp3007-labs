"""
Machine Perception Prac03
Daehwan Yeo

Exercise 1 - Corner Detection

"""

## Harris Detection Method
img = cv2.imread("prac03ex01img01.png")

def harris_corner_detection(img, ):
    # Convert the image to grayscale if it's not already
    if len(img.shape) == 3:
        gray =
    else:
        gray =

    # Detect Harris corners
    harris_corners =

    # Create a copy of the image to mark the corners
    img_harris =

    # Mark detected corners as red stars

    return img_harris

# Harris corners example
harris_img = harris_corner_detection(img=img, )

# Convert BGR to RGB for displaying with matplotlib
plt.imshow(cv2.cvtColor(harris_img, cv2.COLOR_BGR2RGB))
plt.title('Harris Corners')