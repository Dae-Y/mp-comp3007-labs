"""
Machine Perception Prac03
Daehwan Yeo

Exercise 1 - Corner Detection
Harris + Shi-Tomasi with tunable parameters
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Harris Corner Detection
# -----------------------------
def harris_corner_detection(img,
                            blockSize=3,    # neighbourhood for structure tensor (2/3/5)
                            ksize=3,        # Sobel kernel size (3/5/7)
                            k=0.04,         # Harris free parameter (0.04–0.06 typical)
                            thresh_ratio=0.02):  # fraction of max R used as threshold
    """
    Returns a copy of the input image with Harris corners marked as red stars.
    """
    # Ensure grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # float32 required
    gray_f = np.float32(gray)

    # Harris response
    R = cv2.cornerHarris(gray_f, blockSize=blockSize, ksize=ksize, k=k)
    R = cv2.dilate(R, None)  # make responses more visible

    # Threshold
    R_max = R.max()
    T = thresh_ratio * R_max

    # Mark corners
    out = img.copy()
    ys, xs = np.where(R > T)
    for (x, y) in zip(xs, ys):
        cv2.drawMarker(out, (x, y), color=(0, 0, 255), markerType=cv2.MARKER_STAR,
                       markerSize=8, thickness=1, line_type=cv2.LINE_AA)
    return out


# -----------------------------
# Shi–Tomasi Corner Detection
# -----------------------------
def shi_tomasi_corner_detection(img,
                                maxCorners=200,      # 0 => all; else strongest N
                                qualityLevel=0.01,   # min accepted quality (relative)
                                minDistance=10,      # min distance between corners
                                blockSize=3,         # neighbourhood for covariance
                                useHarrisDetector=False, k=0.04):
    """
    Returns a copy of the input image with Shi-Tomasi corners marked as green circles.
    """
    # Ensure grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Detect corners
    corners = cv2.goodFeaturesToTrack(gray,
                                     maxCorners=maxCorners,
                                     qualityLevel=qualityLevel,
                                     minDistance=minDistance,
                                     blockSize=blockSize,
                                     useHarrisDetector=useHarrisDetector,
                                     k=k)
    out = img.copy()
    if corners is not None:
        corners = np.int64(corners)  # Corrected line: 'np.int0' changed to 'np.int64'
        for c in corners:
            x, y = c.ravel()
            cv2.circle(out, (x, y), radius=4, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    return out


# -----------------------------
# Run on the images
# -----------------------------
# Harris on ideal checkerboard
img_harris_src = cv2.imread("prac03ex01img01.png")
harris_img = harris_corner_detection(
    img=img_harris_src,
    blockSize=3,  # try 2, 3, 5
    ksize=3,      # try 3, 5, 7
    k=0.04,       # try 0.04–0.06
    thresh_ratio=0.02  # try 0.01–0.05 (lower -> more points)
)

plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(harris_img, cv2.COLOR_BGR2RGB))
plt.title('Harris Corners')
plt.axis('off')
plt.tight_layout()
plt.savefig("ex01_Harris_result.png", dpi=300, bbox_inches='tight')
print("Saved: ex01_Harris_result.png")
plt.close()

# Shi–Tomasi on contrast/blurred checkerboard (or any test image)
img_shi_src = cv2.imread("prac03ex01img02.png")
shi_tomasi_img = shi_tomasi_corner_detection(
    img=img_shi_src,
    maxCorners=400,      # increase if you want more points
    qualityLevel=0.01,   # raise to 0.02–0.05 for stricter corners
    minDistance=8,       # increase to space them out more
    blockSize=3
)

plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(shi_tomasi_img, cv2.COLOR_BGR2RGB))
plt.title('Shi-Tomasi Corners')
plt.axis('off')
plt.tight_layout()
plt.savefig("ex01_Shi_result.png", dpi=300, bbox_inches='tight')
print("Saved: ex01_Shi_result.png")
plt.close()
