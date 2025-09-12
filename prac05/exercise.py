"""
Machine perception prac5 - Machine Learning part 1
Daehwan Yeo

Exercise - Image Classification with SVM and Nearest Neighbor Methods

- Uses OpenCV's ROW_SAMPLE layout.
- Converts each 20x20 digit into a row vector of length 400 (raw pixels) or HOG features.
- Evaluates kNN (k=1..7) and Linear C-SVM (C on log scale).
- Saves figures and a text summary to results/result.txt

results/result.txt — a full text log (error rates, confusion matrices)
results/knn_0_1.png, results/knn_3_8.png, results/knn_all.png — kNN error curves (raw pixels)
results/knn_hog_3_8.png, results/knn_hog_all.png — kNN error curves (HOG)
results/svm_linear.png — SVM error vs C plot
"""

import os
import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

# -----------------------------
# Utilities
# -----------------------------
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
RESULT_TXT = os.path.join(RESULT_DIR, "result.txt")

def save_fig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
    print(f"[Saved] {path}")
    plt.close()

def confusion_matrix_np(true_labels, pred_labels, num_classes=10):
    """
    true_labels, pred_labels: 1D int arrays of same length
    returns: (num_classes x num_classes) matrix where [i,j] counts true=i, pred=j
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(true_labels, pred_labels):
        cm[int(t), int(p)] += 1
    return cm

def error_rate_from_preds(pred, true):
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)
    return 100.0 * (1.0 - np.mean(pred == true))

def split_digits_image_to_train_test(gray):
    """
    digits.png has 50 rows x 100 cols of 20x20 cells = 5000 images (500 per class).
    We split into train (first 50 cols) and test (last 50 cols).
    Returns:
        train (2500 x 400 float32), test (2500 x 400 float32),
        train_labels (2500 x 1 int32), test_labels (2500 x 1 int32)
    """
    # 50x100 grid of 20x20 patches
    cells = [np.hsplit(r, 100) for r in np.vsplit(gray, 50)]
    x = np.array(cells)  # shape (50, 100, 20, 20)

    # ROW_SAMPLE layout: flatten each 20x20 to 400
    train = x[:, :50].reshape(-1, 400).astype(np.float32)   # (2500, 400)
    test  = x[:, 50:100].reshape(-1, 400).astype(np.float32)  # (2500, 400)

    # labels: 0..9 repeated 250 times per class for train, same for test
    train_labels = np.repeat(np.arange(10), 250).astype(np.int32).reshape(-1, 1)
    test_labels = train_labels.copy()

    return train, train_labels, test, test_labels

def filter_binary_classes(data, labels, a, b):
    """
    Keep only class a and b from (data, labels). Returns filtered (data, labels).
    Labels remain original values a or b.
    """
    idx = np.where((labels.reshape(-1) == a) | (labels.reshape(-1) == b))[0]
    return data[idx], labels[idx]

def calculate_error_rate(train_data, train_labels, test_data, test_labels, k_values):
    """
    Train kNN on train_data and print error rate for each k in k_values.
    Labels must be (N,1) int32 for OpenCV ml.
    """
    # Create and train once (OpenCV's kNN ignores k at training; k is used in findNearest)
    knn = cv2.ml.KNearest_create()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

    for k in k_values:
        _, result, _, _ = knn.findNearest(test_data, k=k)
        err = error_rate_from_preds(result, test_labels)
        print(f"kNN | k={k} | Test error rate = {err:.2f}%")

def plot_knn_curve(k_list, errors, title, save_path):
    plt.figure()
    plt.plot(k_list, errors, marker="o")
    plt.xticks(k_list)
    plt.xlabel("k")
    plt.ylabel("Error rate (%)")
    plt.title(title)
    plt.grid(alpha=0.3)
    save_fig(save_path)

def compute_knn_errors(train_data, train_labels, test_data, test_labels, k_values):
    knn = cv2.ml.KNearest_create()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    errs = []
    for k in k_values:
        _, result, _, _ = knn.findNearest(test_data, k=k)
        err = error_rate_from_preds(result, test_labels)
        errs.append(err)
        print(f"kNN | k={k} | Test error rate = {err:.2f}%")
    return errs

# -----------------------------
# HOG features (optional exercise 2)
# -----------------------------
def build_hog():
    # HOG over 20x20 window. Blocks 10x10, stride 5x5, cells 10x10, 9 bins.
    return cv2.HOGDescriptor(_winSize=(20, 20),
                             _blockSize=(10, 10),
                             _blockStride=(5, 5),
                             _cellSize=(10, 10),
                             _nbins=9)

def extract_hog_features_from_split(gray):
    """
    From the digits image, produce HOG features for train/test.
    Returns: train (2500 x D), train_labels (2500 x 1),
             test  (2500 x D), test_labels  (2500 x 1)
    """
    hog = build_hog()
    cells = [np.hsplit(r, 100) for r in np.vsplit(gray, 50)]  # (50,100,20,20)

    def hog_map(cell_rows):
        feats = []
        for row in cell_rows:
            for patch in row:
                # HOG expects 8-bit single-channel image of size winSize
                f = hog.compute(patch.astype(np.uint8))  # (D,1)
                feats.append(f.reshape(-1))
        return np.array(feats, dtype=np.float32)

    # train (first 50 columns), test (last 50)
    train_feats = hog_map([r[:50] for r in cells])
    test_feats  = hog_map([r[50:100] for r in cells])

    train_labels = np.repeat(np.arange(10), 250).astype(np.int32).reshape(-1, 1)
    test_labels  = train_labels.copy()

    return train_feats, train_labels, test_feats, test_labels

# -----------------------------
# SVM (Exercise 3)
# -----------------------------
def run_svm_linear(train, train_labels, test, test_labels, C_values, num_classes=10):
    """
    Train Linear C-SVM for each C in C_values, report train/test error and confusion matrices.
    Returns (C_values, test_errors) for plotting.
    """
    test_errors = []

    for C in C_values:
        svm = cv2.ml.SVM_create()
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setC(float(C))
        # Termination: max 100 iters or eps
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-8))

        svm.train(train, cv2.ml.ROW_SAMPLE, train_labels)

        train_pred = svm.predict(train)[1].astype(np.int32).reshape(-1)
        test_pred  = svm.predict(test)[1].astype(np.int32).reshape(-1)

        tr_err = error_rate_from_preds(train_pred, train_labels.reshape(-1))
        te_err = error_rate_from_preds(test_pred, test_labels.reshape(-1))
        test_errors.append(te_err)

        tr_cm = confusion_matrix_np(train_labels.reshape(-1), train_pred, num_classes=num_classes)
        te_cm = confusion_matrix_np(test_labels.reshape(-1), test_pred, num_classes=num_classes)

        print(f"SVM Linear | C={C:.5g} | Train err={tr_err:.2f}% | Test err={te_err:.2f}%")
        print("Train Confusion Matrix:\n", tr_cm)
        print("Test  Confusion Matrix:\n", te_cm)
        print("-" * 60)

    return list(C_values), test_errors

def plot_svm_curve(C_values, test_errors, title, save_path):
    plt.figure()
    plt.semilogx(C_values, test_errors, marker="o")
    plt.xlabel("C (log scale)")
    plt.ylabel("Test error rate (%)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    save_fig(save_path)

# -----------------------------
# Main Exercises
# -----------------------------
def exercise_knn_raw(gray):
    print("\n=== Exercise 1: kNN on raw pixels (ROW_SAMPLE = 20x20 -> 400) ===")
    train, train_labels, test, test_labels = split_digits_image_to_train_test(gray)

    # Binary 0 vs 1 (500 each -> 250 train + 250 test per digit)
    print("\n[Binary] Digits 0 vs 1")
    tr_01, ytr_01 = filter_binary_classes(train, train_labels, 0, 1)
    te_01, yte_01 = filter_binary_classes(test,  test_labels,  0, 1)
    k_values = list(range(1, 8))
    errs_01 = compute_knn_errors(tr_01, ytr_01, te_01, yte_01, k_values)
    plot_knn_curve(k_values, errs_01, "kNN Error (0 vs 1, raw pixels)", os.path.join(RESULT_DIR, "knn_0_1.png"))

    # Binary 3 vs 8
    print("\n[Binary] Digits 3 vs 8")
    tr_38, ytr_38 = filter_binary_classes(train, train_labels, 3, 8)
    te_38, yte_38 = filter_binary_classes(test,  test_labels,  3, 8)
    errs_38 = compute_knn_errors(tr_38, ytr_38, te_38, yte_38, k_values)
    plot_knn_curve(k_values, errs_38, "kNN Error (3 vs 8, raw pixels)", os.path.join(RESULT_DIR, "knn_3_8.png"))

    # All digits
    print("\n[Multiclass] All digits 0..9")
    errs_all = compute_knn_errors(train, train_labels, test, test_labels, k_values)
    plot_knn_curve(k_values, errs_all, "kNN Error (All digits, raw pixels)", os.path.join(RESULT_DIR, "knn_all.png"))

def exercise_knn_hog(gray):
    print("\n=== Exercise 2: kNN with HOG features ===")
    train, train_labels, test, test_labels = extract_hog_features_from_split(gray)

    # Binary 3 vs 8
    print("\n[Binary] Digits 3 vs 8 (HOG)")
    tr_38, ytr_38 = filter_binary_classes(train, train_labels, 3, 8)
    te_38, yte_38 = filter_binary_classes(test,  test_labels,  3, 8)
    k_values = list(range(1, 8))
    errs_38 = compute_knn_errors(tr_38, ytr_38, te_38, yte_38, k_values)
    plot_knn_curve(k_values, errs_38, "kNN Error (3 vs 8, HOG)", os.path.join(RESULT_DIR, "knn_hog_3_8.png"))

    # All digits
    print("\n[Multiclass] All digits 0..9 (HOG)")
    errs_all = compute_knn_errors(train, train_labels, test, test_labels, k_values)
    plot_knn_curve(k_values, errs_all, "kNN Error (All digits, HOG)", os.path.join(RESULT_DIR, "knn_hog_all.png"))

def exercise_svm(gray):
    print("\n=== Exercise 3: Linear SVM (C-SVM) ===")
    train, train_labels, test, test_labels = split_digits_image_to_train_test(gray)
    # Normalize pixel values to [0,1] for SVM stability
    train = (train / 255.0).astype(np.float32)
    test  = (test  / 255.0).astype(np.float32)

    # C from very small to large (log scale)
    C_values = np.logspace(-3, 2, num=6)  # 0.001 ... 100
    C_list, test_errs = run_svm_linear(train, train_labels, test, test_labels, C_values, num_classes=10)
    plot_svm_curve(C_list, test_errs, "Linear SVM Test Error vs C", os.path.join(RESULT_DIR, "svm_linear.png"))

# -----------------------------
# Entry point
# -----------------------------
def main():
    img_path = "digits.png"
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Cannot find {img_path} in {os.getcwd()}")
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError("cv2.imread failed to load digits.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Capture all prints to a file AND show on console
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    with open(RESULT_TXT, "w", encoding="utf-8") as f:
        tee = Tee(sys.stdout, f)
        with redirect_stdout(tee):
            print("=== Machine perception prac5: Results ===")
            print(f"Working directory: {os.getcwd()}")
            print(f"digits.png shape: {img.shape}")
            print("-" * 60)

            # kNN raw pixels
            exercise_knn_raw(gray)
            print("-" * 60)

            # kNN HOG features (optional, but part of the practical brief)
            exercise_knn_hog(gray)
            print("-" * 60)

            # SVM (Linear C-SVM)
            exercise_svm(gray)
            print("-" * 60)
            print(f"All results & figures saved under: {RESULT_DIR}")

if __name__ == "__main__":
    main()
