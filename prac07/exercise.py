"""
Machine perception prac7 - Object Detection
Daehwan Yeo

What we cover:
1) Template Matching (cv2.matchTemplate) — fast pattern search; brittle to scale/rotation/illumination.
2) R-CNN-style pipeline — Selective Search → classify proposals with ResNet50 → NMS.
   Slow because we run CNN per-region; NMS removes redundant overlaps.
3) Faster R-CNN (Detectron2) — RPN shares conv features; proposals are cheap + end-to-end training.
"""

# =========================
# Exercise 1 — Template Matching
# =========================
# Minimal steps:
# - Read image/template
# - matchTemplate (e.g., TM_CCOEFF_NORMED)
# - threshold → draw hits
# - (After Ex.2) You can run NMS on matched boxes using the NMS below

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load both images in color
image = cv2.imread('fruits.png', cv2.IMREAD_COLOR)
template = cv2.imread('orange.png', cv2.IMREAD_COLOR)

# Convert images from BGR to RGB color space
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None
template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB) if template is not None else None

# Get the dimensions of the template
h, w = template_rgb.shape[:2]

# Perform template matching
method = cv2.TM_CCOEFF_NORMED
result = cv2.matchTemplate(image_rgb, template_rgb, method)

# Define an acceptable threshold (tune as needed)
threshold = 0.80

# Find all matches above the threshold
ys, xs = np.where(result >= threshold)
locations = list(zip(xs, ys))  # (x, y) list

# Draw rectangles around all matches
for (x, y) in locations:
    color = tuple(np.random.randint(0, 255, 3).tolist())
    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), color, 2)

# Display + save the result
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.title('Template Matching Result')
plt.axis('off')
plt.savefig('ex1_result.png', bbox_inches='tight', dpi=150)
plt.show()

"""
Template Matching — Short take:
+ Simple & fast; OK when the object is same scale/orientation/lighting.
- Fails under scale/rotation/illumination changes; may produce many overlapping hits.
Tip: After you implement NMS (Ex.2), you can deduplicate overlapping matches by running NMS on
the match locations using the response values as scores.
"""


# =========================
# Exercise 2 — R-CNN-Style Detection
# =========================
# Pipeline:
# 1) Region proposals via Selective Search
# 2) Classify each proposal with pre-trained ResNet50 (ImageNet)
# 3) Non-Maximum Suppression (NMS) to prune overlaps
#
# Note: This is slow because we run the CNN for hundreds/thousands of regions.

import random
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm.notebook import tqdm

def selective_search(image):
    """
    Perform selective search on the input image to generate region proposals.
    Returns: array of [x, y, w, h]
    """
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()  # use .switchToSelectiveSearchQuality() for more (slower) proposals
    boxes = ss.process()
    return boxes

def visualise_random_proposals(image, boxes, num_show=20):
    """
    Draw a random subset of proposals on a copy of the image (expects RGB).
    """
    output = image.copy()
    selected_boxes = random.sample(list(boxes), min(len(boxes), num_show))
    for x, y, w, h in selected_boxes:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
    return output

# Load the image and convert to RGB space
image_path = '/content/computer.jpeg'
image_bgr = cv2.imread(image_path)
assert image_bgr is not None, f"Could not load image at {image_path}"
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Run selective search to get region proposals
boxes = selective_search(image)

# Print info
print(f"Number of region proposals: {len(boxes)}")
print(f"Shape of boxes: {boxes.shape}")
print(f"First box: {boxes[0]}")  # [x, y, w, h]

# Visualise some proposals
result = visualise_random_proposals(image, boxes, num_show=40)

# Display + save
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(result)
ax.set_title('Selective Search — Random Proposals')
ax.axis('off')
fig.savefig('ex2_proposals.png', bbox_inches='tight', dpi=150)
plt.show()


# ---------- NMS Utilities ----------

def calculate_iou(box1, box2):
    """
    IoU for boxes in (x, y, w, h).
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    inter_w = max(0, min(x1_max, x2_max) - max(x1, x2))
    inter_h = max(0, min(y1_max, y2_max) - max(y1, y2))
    inter = inter_w * inter_h

    area1 = max(0, w1) * max(0, h1)
    area2 = max(0, w2) * max(0, h2)
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union

def non_max_suppression(boxes, scores, iou_threshold):
    """
    NMS over (x, y, w, h) boxes.
    Returns: indices to keep.
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=float)
    scores = np.array(scores, dtype=float)
    order = np.argsort(scores)[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        rest = order[1:]
        ious = np.array([calculate_iou(boxes[i], boxes[j]) for j in rest])
        remain = rest[ious <= iou_threshold]
        order = remain

    return keep


# ---------- Classify proposals with ResNet50 ----------

def get_region_scores_and_classes(image_rgb, boxes, min_size=20):
    """
    Classify each region with ResNet50 (ImageNet). Returns (scores, class_names).
    For tiny regions (< min_size), returns background with score 0.0 to skip noise.
    """
    scores, classes = [], []
    for box in tqdm(boxes, desc="Processing regions"):
        x, y, w, h = map(int, box)
        if w < min_size or h < min_size:
            scores.append(0.0)
            classes.append('background')
            continue

        # Crop & preprocess (keep RGB; preprocess_input handles mean/scale)
        region = image_rgb[y:y+h, x:x+w]
        if region.size == 0:
            scores.append(0.0)
            classes.append('background')
            continue

        region = cv2.resize(region, (224, 224))
        region = img_to_array(region)
        region = np.expand_dims(region, axis=0)
        region = preprocess_input(region)

        preds = model.predict(region, verbose=0)
        cls_id, cls_name, score = decode_predictions(preds, top=1)[0][0]
        scores.append(float(score))
        classes.append(cls_name)

    return np.array(scores, dtype=float), classes

def visualise_best_proposals(image_rgb, boxes, scores, classes, targets, num_show=1):
    """
    Draw top scoring proposals per target class on a copy of the image (RGB).
    """
    output = image_rgb.copy()
    color_map = {t: tuple(np.random.randint(0, 255, 3).tolist()) for t in targets}

    for target in targets:
        idxs = [i for i, c in enumerate(classes) if c.lower() == target.lower()]
        if not idxs:
            continue
        target_scores = scores[idxs]
        target_boxes = np.array(boxes)[idxs]

        order = np.argsort(target_scores)[::-1][:num_show]
        for k in order:
            x, y, w, h = map(int, target_boxes[k])
            s = target_scores[k]
            color = color_map[target]
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)

            label = f"{target}: {s:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x, y - th - bl), (x + tw, y), color, -1)
            cv2.putText(output, label, (x, y - bl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return output


# Load ImageNet pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Get scores and class predictions for each region proposal using ResNet50
scores, classes = get_region_scores_and_classes(image, boxes)

# Perform NMS
iou_threshold = 0.30
kept_indices = non_max_suppression(boxes, scores, iou_threshold)

# Get the boxes, scores, and classes after NMS
nms_boxes = np.array(boxes)[kept_indices]
nms_scores = scores[kept_indices]
nms_classes = [classes[i] for i in kept_indices]

print(f"Number of boxes before NMS: {len(boxes)}")
print(f"Number of boxes after NMS:  {len(nms_boxes)}")

# Visualise the top NMS proposals for the target classes
targets = ['desktop_computer', 'mouse', 'computer_keyboard']
nms_vis = visualise_best_proposals(image, nms_boxes, nms_scores, nms_classes, targets, num_show=1)

plt.figure(figsize=(10, 10))
plt.imshow(nms_vis)
plt.axis('off')
plt.title('Top Proposals for Target Classes (after NMS)')
plt.savefig('ex2_nms_targets.png', bbox_inches='tight', dpi=150)
plt.show()

"""
Why slow?
R-CNN classifies each region separately with a full CNN forward pass → massive redundancy.
Fast R-CNN: run CNN once on the full image, then RoI Pool on feature map → big speedup.
Faster R-CNN: add an RPN that shares features to produce proposals → near real-time + better AP.
"""


# =========================
# Exercise 3 — Faster R-CNN with Detectron2
# =========================
# Keep your original install cell (Colab-style). In scripts/notebooks, run as a shell cell.

import sys, os, distutils.core
# Note: This is a faster way to install detectron2 in Colab, but it does not include all functionalities (e.g. compiled operators).
# See https://detectron2.readthedocs.io/tutorials/install.html for full installation instructions
!git clone 'https://github.com/facebookresearch/detectron2'
dist = distutils.core.run_setup("./detectron2/setup.py")
!python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])}
sys.path.insert(0, os.path.abspath('./detectron2'))

"""
Detectron2 (Short recipe):
1) Load pre-trained Faster R-CNN from model zoo; run inference on a test image; visualise.
2) Register your COCO-format blood cell dataset (train/val) via DatasetCatalog/MetadataCatalog.
3) Configure model (e.g., R50-FPN), set SOLVER/LR/IMS_PER_BATCH/MAX_ITER, and TRAIN.
4) Track AP/loss (TensorBoard). Improve via:
   - Lower/higher NMS IoU threshold for more/less boxes
   - Stronger backbones (R101, ResNeXt), longer schedule, data augmentation
   - Class-imbalance handling (e.g., focal loss proxy via RetinaNet, or class-balanced sampling/weights)
Goal: lift WBC AP (minority class) with augmentation + sampling or loss reweighting.
"""
