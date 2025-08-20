'''
Machine perception prac5 - Machine Learning part 1
Daehwan Yeo

Exercise - Image Classification with SVM and Nearest Neighbor Methods

To familiarize with different classification methods discussed in the lectures.
In this practical, k-Nearest Neighbours (k-NN) and Support Vector Machines (SVM) classification methods will be used.
'''

# Importing some useful packages
import numpy as np # Numpy library provides various useful functions and operators for scientific computing
import cv2 as cv   # OpenCV is a key library that provides various useful functions for computer vision
import os          # Optional
import glob        # Optional
import sys
from matplotlib import pyplot as plt

# --- Load image ---
digits = cv.imread("digits.png")
if digits is None:
    print("Error: Could not load the image 'digits.png'. Please ensure it's in the same directory.")
    sys.exit(1)

gray = cv.cvtColor(digits, cv.COLOR_BGR2GRAY)
height, width = gray.shape
print("The image height is: ", height)
print("The image width is: ", width)

# Split into 50x100 = 5000 cells
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
x = np.array(cells)  # (50, 100, 20, 20)

# Flatten to (5000, 400)
train_features = x.reshape(-1, 400).astype(np.float32)

# Labels (0-9 repeated 500 times each)
k = np.arange(10, dtype=np.int32)
train_labels = np.repeat(k, 500)[:, np.newaxis].astype(np.float32)

# ----------------------------------------------------------------
# Save a sample grid of digits (first 100 digits)
sample_grid = gray[:200, :2000]  # top-left corner (10x10 digits)
plt.figure(figsize=(10, 2))
plt.imshow(sample_grid, cmap="gray")
plt.axis("off")
plt.title("Sample digits from dataset")
plt.savefig("sample_digits.png")
plt.close()
# ----------------------------------------------------------------

def classify_and_evaluate(features, labels, class_1, class_2, split_ratio=0.5, save_prefix="output"):
    """
    Performs binary classification on two specified classes using k-NN and SVM.
    
    Args:
        features (np.array): The feature matrix of all digits.
        labels (np.array): The labels for all digits (Nx1 float32).
        class_1 (int): The first digit class to be used for classification.
        class_2 (int): The second digit class to be used for classification.
        split_ratio (float): The ratio of data to use for training.
        save_prefix (str): Prefix for saving plot images.
        
    Returns:
        None: Prints the classification results and error rates.
    """
    # Convert classes to float32 to match labels dtype
    c1 = np.float32(class_1)
    c2 = np.float32(class_2)

    # Select data for the two specified classes
    class_1_indices = np.where(labels.ravel() == c1)[0]
    class_2_indices = np.where(labels.ravel() == c2)[0]
    
    # Combine the features and labels for the selected classes
    binary_features = np.concatenate((features[class_1_indices], features[class_2_indices]), axis=0)
    binary_labels = np.concatenate((labels[class_1_indices], labels[class_2_indices]), axis=0)

    # Split the data into training and testing sets
    split_index = int(len(binary_features) * split_ratio)

    train_data = np.concatenate((binary_features[:split_index], binary_features[500:500+split_index]), axis=0)
    test_data  = np.concatenate((binary_features[split_index:500], binary_features[500+split_index:1000]), axis=0)
    
    train_labels_binary = np.concatenate((binary_labels[:split_index], binary_labels[500:500+split_index]), axis=0)
    test_labels_binary  = np.concatenate((binary_labels[split_index:500], binary_labels[500+split_index:1000]), axis=0)
    
    print(f"\n--- Binary Classification: Digits {class_1} and {class_2} ---")
    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of testing samples: {len(test_data)}")

    # --- k-NN Classifier ---
    knn = cv.ml.KNearest_create()
    knn.train(train_data, cv.ml.ROW_SAMPLE, train_labels_binary)
    
    # Set k=1 for the initial test
    k_val = 1
    ret, result_knn, neighbors, dist = knn.findNearest(test_data, k_val)
    
    # Calculate k-NN error
    matches_knn = (result_knn == test_labels_binary)
    correct_knn = int(np.count_nonzero(matches_knn))
    incorrect_knn = len(test_data) - correct_knn
    error_rate_knn = (incorrect_knn / len(test_data)) * 100.0
    
    print("\n[k-NN Classifier (k=1)]")
    print(f"Incorrectly classified testing samples: {incorrect_knn}")
    print(f"Testing error rate: {error_rate_knn:.2f}%")

    # --- SVM Classifier ---
    svm = cv.ml.SVM_create()
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setC(1.0)
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 100, 1e-8))
    
    svm.train(train_data, cv.ml.ROW_SAMPLE, train_labels_binary)
    
    # Test on testing data
    ret, result_svm_test = svm.predict(test_data)
    
    # Calculate SVM test error
    matches_svm_test = (result_svm_test == test_labels_binary)
    correct_svm_test = int(np.count_nonzero(matches_svm_test))
    incorrect_svm_test = len(test_data) - correct_svm_test
    error_rate_svm_test = (incorrect_svm_test / len(test_data)) * 100.0
    
    # Test on training data
    ret, result_svm_train = svm.predict(train_data)
    
    # Calculate SVM training error
    matches_svm_train = (result_svm_train == train_labels_binary)
    incorrect_svm_train = int(np.count_nonzero(~matches_svm_train))
    error_rate_svm_train = (incorrect_svm_train / len(train_data)) * 100.0
    
    print("\n[SVM Classifier (C=1)]")
    print(f"Incorrectly classified testing samples: {incorrect_svm_test}")
    print(f"Testing error rate: {error_rate_svm_test:.2f}%")
    print(f"Incorrectly classified training samples: {incorrect_svm_train}")
    print(f"Training error rate: {error_rate_svm_train:.2f}%")

    # --- Save some example results as PNG ---
    # Show 10 test digits with true vs predicted labels (SVM)
    fig, axes = plt.subplots(1, 10, figsize=(12, 2))
    for idx, ax in enumerate(axes):
        img = test_data[idx].reshape(20, 20)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(f"T:{int(test_labels_binary[idx,0])}\nP:{int(result_svm_test[idx,0])}")
    plt.suptitle(f"SVM predictions for digits {class_1} vs {class_2}")
    plt.savefig(f"{save_prefix}_svm_predictions.png")
    plt.close()

# --- Run classifiers ---
classify_and_evaluate(train_features, train_labels, 0, 1, save_prefix="digits01")
classify_and_evaluate(train_features, train_labels, 3, 8, save_prefix="digits38")

def tune_parameters(features, labels):
    """
    Tunes the k and C parameters for k-NN and SVM on all digits.
    
    Args:
        features (np.array): The feature matrix of all digits.
        labels (np.array): The labels for all digits.
    """
    # Split the data for multi-category classification (all digits)
    train_data_all = features[2500:]
    test_data_all  = features[:2500]
    train_labels_all = labels[2500:]
    test_labels_all  = labels[:2500]
    
    # --- k-NN: Varying k ---
    k_values = range(1, 8)
    knn_errors = []
    
    knn = cv.ml.KNearest_create()
    knn.train(train_data_all, cv.ml.ROW_SAMPLE, train_labels_all)

    print("\n--- Tuning k for k-NN (Multi-category Classification) ---")
    for k_val in k_values:
        ret, result, neighbors, dist = knn.findNearest(test_data_all, k_val)
        matches = (result == test_labels_all)
        incorrect = int(np.count_nonzero(~matches))
        error_rate = (incorrect / len(test_data_all)) * 100.0
        knn_errors.append(error_rate)
        print(f"k = {k_val}: Error Rate = {error_rate:.2f}%")

    # Plotting the results
    plt.figure()
    plt.plot(list(k_values), knn_errors, marker='o')
    plt.title('k-NN Error Rate vs. k')
    plt.xlabel('k')
    plt.ylabel('Error Rate (%)')
    plt.grid(True)
    plt.savefig('knn_error_vs_k.png')
    plt.close()

    # --- SVM: Varying C ---
    c_values = np.logspace(-3, 3, 7)  # from 1e-3 to 1e3
    svm_errors = []

    print("\n--- Tuning C for SVM (Multi-category Classification) ---")
    for C in c_values:
        svm = cv.ml.SVM_create()
        svm.setKernel(cv.ml.SVM_LINEAR)
        svm.setType(cv.ml.SVM_C_SVC)
        svm.setC(float(C))
        svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 100, 1e-8))
        
        svm.train(train_data_all, cv.ml.ROW_SAMPLE, train_labels_all)
        
        ret, result = svm.predict(test_data_all)
        matches = (result == test_labels_all)
        incorrect = int(np.count_nonzero(~matches))
        error_rate = (incorrect / len(test_data_all)) * 100.0
        svm_errors.append(error_rate)
        print(f"C = {C:.3f}: Error Rate = {error_rate:.2f}%")

    # Plotting the results
    plt.figure()
    plt.plot(c_values, svm_errors, marker='o')
    plt.xscale('log')
    plt.title('SVM Error Rate vs. C')
    plt.xlabel('C (log scale)')
    plt.ylabel('Error Rate (%)')
    plt.grid(True)
    plt.savefig('svm_error_vs_c.png')
    plt.close()

# Run the parameter tuning function on all digits
tune_parameters(train_features, train_labels)

# ======================================================================================================================
# --- NEW SECTION: CLASSIFICATION WITH HOG FEATURES ---
# ======================================================================================================================

def classify_with_hog_features(features_2d, labels, class_1, class_2, save_prefix="hog"):
    """
    Extracts HOG features and performs binary classification for the two given classes.
    
    Args:
        features_2d (np.array): The original 2D image data of shape (N, 20, 20).
        labels (np.array): The labels for all digits (Nx1 float32).
        class_1 (int): The first digit class to be used.
        class_2 (int): The second digit class to be used.
        save_prefix (str): Prefix for saving plot images.
    """
    print(f"\n--- Classification with HOG Features: Digits {class_1} and {class_2} ---")
    
    # Initialize the HOG descriptor for 20x20 images
    hog = cv.HOGDescriptor(_winSize=(20, 20),
                           _blockSize=(10, 10),
                           _blockStride=(5, 5),
                           _cellSize=(10, 10),
                           _nbins=9)
    
    # Extract HOG features for all samples
    hog_features_list = []
    for i in range(features_2d.shape[0]):
        # OpenCV expects a 2D uint8 image for HOG
        img_u8 = features_2d[i, :, :].astype(np.uint8)
        hog_feature = hog.compute(img_u8)  # (num_features, 1)
        hog_features_list.append(hog_feature)
    
    hog_features = np.array(hog_features_list).reshape(features_2d.shape[0], -1).astype(np.float32)
    print(f"Shape of HOG feature matrix: {hog_features.shape}")

    # Select data for the two specified classes
    class_1_indices = np.where(labels.ravel() == class_1)[0]
    class_2_indices = np.where(labels.ravel() == class_2)[0]
    
    binary_hog_features = np.concatenate((hog_features[class_1_indices], hog_features[class_2_indices]), axis=0)
    binary_labels = np.concatenate((labels[class_1_indices], labels[class_2_indices]), axis=0)

    # Split the HOG data into training and testing sets (50/50 split)
    split_index = int(len(binary_hog_features) / 2)
    
    train_data_hog = np.concatenate((binary_hog_features[:split_index], binary_hog_features[500:500+split_index]), axis=0)
    test_data_hog = np.concatenate((binary_hog_features[split_index:500], binary_hog_features[500+split_index:1000]), axis=0)
    
    train_labels_hog = np.concatenate((binary_labels[:split_index], binary_labels[500:500+split_index]), axis=0)
    test_labels_hog = np.concatenate((binary_labels[split_index:500], binary_labels[500+split_index:1000]), axis=0)

    # --- k-NN Classifier with HOG ---
    knn_hog = cv.ml.KNearest_create()
    knn_hog.train(train_data_hog, cv.ml.ROW_SAMPLE, train_labels_hog)
    
    ret, result_knn_hog, neighbors, dist = knn_hog.findNearest(test_data_hog, k=1)
    
    matches_knn_hog = result_knn_hog == test_labels_hog
    incorrect_knn_hog = int(np.count_nonzero(matches_knn_hog == False))
    error_rate_knn_hog = (incorrect_knn_hog / len(test_data_hog)) * 100.0
    
    print("\n[k-NN Classifier (k=1) with HOG]")
    print(f"Incorrectly classified testing samples: {incorrect_knn_hog}")
    print(f"Testing error rate: {error_rate_knn_hog:.2f}%")

    # --- SVM Classifier with HOG ---
    svm_hog = cv.ml.SVM_create()
    svm_hog.setKernel(cv.ml.SVM_LINEAR)
    svm_hog.setType(cv.ml.SVM_C_SVC)
    svm_hog.setC(1.0)
    svm_hog.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 100, 1e-8))
    
    svm_hog.train(train_data_hog, cv.ml.ROW_SAMPLE, train_labels_hog)
    
    ret, result_svm_hog = svm_hog.predict(test_data_hog)
    
    matches_svm_hog = result_svm_hog == test_labels_hog
    incorrect_svm_hog = int(np.count_nonzero(matches_svm_hog == False))
    error_rate_svm_hog = (incorrect_svm_hog / len(test_data_hog)) * 100.0
    
    print("\n[SVM Classifier (C=1) with HOG]")
    print(f"Incorrectly classified testing samples: {incorrect_svm_hog}")
    print(f"Testing error rate: {error_rate_svm_hog:.2f}%")

    # --- Save some HOG test results ---
    fig, axes = plt.subplots(1, 10, figsize=(12, 2))
    for idx, ax in enumerate(axes):
        # We can't back-project HOG easily, so just show grayscale original
        
        # Get the original index for the test data
        test_labels_flat = test_labels_hog.ravel()
        original_indices = np.where(labels.ravel() == class_1)[0].tolist() + np.where(labels.ravel() == class_2)[0].tolist()
        
        # The test_data_hog is the second half of the concatenated binary data
        original_idx = 500 + idx
        if int(test_labels_flat[idx]) == class_2:
            original_idx = 500 + 500 + idx
            
        img_original = features_2d[original_indices[original_idx]]
        
        ax.imshow(img_original, cmap="gray")
        ax.axis("off")
        ax.set_title(f"T:{int(test_labels_hog[idx,0])}\nP:{int(result_svm_hog[idx,0])}")
    plt.suptitle(f"HOG+SVM predictions for digits {class_1} vs {class_2}")
    plt.savefig(f"{save_prefix}_svm_predictions.png")
    plt.close()

classify_with_hog_features(x.reshape(5000, 20, 20), train_labels, 3, 8, save_prefix="hog38")