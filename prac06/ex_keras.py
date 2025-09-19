"""
Machine Perception Prac06 - Machine Learning Part 2 (Keras Implementation)
Author: Daehwan Yeo

In this practical, we perform image classification on the CIFAR-10 dataset using deep learning 
approaches implemented in Keras. 

Dataset:
- CIFAR-10 contains 60,000 images (32x32 RGB), across 10 classes.
- Training: 50,000 images (we split into 40,000 train + 10,000 validation)
- Testing: 10,000 images

Exercises:
1. Build a basic CNN with Conv/Pooling layers and compare RMSprop vs Adam.
2. Add data augmentation and dropout, observe performance improvements.
3. Use transfer learning with VGG16 (and optionally ResNet50) as feature extractors.
4. Build a deeper CNN with batch normalization + dropout to maximize performance.

Throughout the experiments:
- Save confusion matrices into `ex_keras/`.
- Write performance summaries and thoughts into keras_result.txt.
"""

# --- Imports ---
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import keras
from keras import layers, regularizers
from tensorflow.keras.applications import VGG16
from keras.applications import ResNet50
from tensorflow.keras.datasets import cifar10

# Ensure results folder exists
os.makedirs("ex_keras", exist_ok=True)


# -----------------------
# Exercise 1: Basic CNN
# -----------------------
def exercise1_keras_basic():
    """
    Basic CNN on CIFAR-10
    Layers: Conv(32) → MaxPool → Conv(64) → MaxPool → Conv(128) → Flatten → Dense(10)
    Compare RMSprop vs Adam optimizers.
    """

    # Load CIFAR-10 data
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images, test_images = train_images.astype("float32")/255, test_images.astype("float32")/255

    # Split train/validation
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, stratify=train_labels, random_state=42
    )

    # Build model
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    # ---- Training with RMSprop ----
    print("Exercise 1 - Training with RMSprop")
    model.compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_images, train_labels,
              epochs=5, batch_size=64,
              validation_data=(val_images, val_labels))
    test_loss_rms, test_acc_rms = model.evaluate(test_images, test_labels)
    print(f"Test accuracy (RMSprop): {test_acc_rms:.3f}")

    # Confusion Matrix (RMSprop)
    train_pred_rms = model.predict(train_images).argmax(axis=1)
    cm_rms = confusion_matrix(train_labels, train_pred_rms)

    # ---- Training with Adam ----
    print("Exercise 1 - Training with Adam")
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_images, train_labels,
              epochs=5, batch_size=64,
              validation_data=(val_images, val_labels))
    test_loss_adam, test_acc_adam = model.evaluate(test_images, test_labels)
    print(f"Test accuracy (Adam): {test_acc_adam:.3f}")

    # Confusion Matrix (Adam)
    train_pred_adam = model.predict(train_images).argmax(axis=1)
    cm_adam = confusion_matrix(train_labels, train_pred_adam)

    # Plot & Save confusion matrices
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_rms, annot=True, fmt="d", cmap="Blues")
    plt.title("Training Confusion Matrix - RMSprop")
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_adam, annot=True, fmt="d", cmap="Blues")
    plt.title("Training Confusion Matrix - Adam")
    plt.savefig("ex_keras/ex1_confusion_matrix.png")


# -------------------------------
# Exercise 2: Data Augmentation
# -------------------------------
def exercise2_keras_augmentation():
    """
    CNN with Data Augmentation (Random Flip) and Dropout (30%).
    """

    # Load + preprocess
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images, test_images = train_images.astype("float32")/255, test_images.astype("float32")/255
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, stratify=train_labels, random_state=42
    )

    # Data augmentation
    data_augmentation = keras.Sequential([layers.RandomFlip("horizontal")])

    # Model
    inputs = keras.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)
    x = layers.Conv2D(32, 3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_images, train_labels,
              epochs=20, batch_size=64,
              validation_data=(val_images, val_labels))
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy (Aug + Dropout): {test_acc:.3f}")

    # Confusion Matrix
    preds = model.predict(test_images).argmax(axis=1)
    cm = confusion_matrix(test_labels, preds)
    np.savetxt("ex_keras/ex2_confusion_matrix.txt", cm, fmt="%d")


# --------------------------------
# Exercise 3: Pre-trained Network
# --------------------------------
def exercise3_keras_pretrained():
    """
    Transfer Learning using VGG16 pre-trained on ImageNet.
    """

    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, stratify=train_labels, random_state=42
    )

    conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
    conv_base.trainable = False  # freeze base

    inputs = keras.Input(shape=(32, 32, 3))
    x = keras.applications.vgg16.preprocess_input(inputs)
    x = conv_base(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    '''
    it will generate ex3_fine_tuning.keras file which is quite large.
    
    How to use it?
    from keras.models import load_model
    model = load_model("ex3_fine_tuning.keras")
    predictions = model.predict(x_test)
    model.evaluate(x_test, y_test)
    model.summary()
    .. etc
    
    model.fit(train_images, train_labels,
              epochs=10, batch_size=32,
              validation_data=(val_images, val_labels),
              callbacks=[keras.callbacks.ModelCheckpoint("ex_keras/ex3_fine_tuning.keras",
                                                         save_best_only=True,
                                                         monitor="val_loss")])
    model = keras.models.load_model("ex_keras/ex3_fine_tuning.keras")
    '''
    model.fit(train_images, train_labels,
              epochs=10, batch_size=32,
              validation_data=(val_images, val_labels))

    # Evaluate on test data
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy (VGG16): {test_acc:.3f}")
    
# -------------------------------------
# Exercise 4: Advanced CNN (from scratch)
# -------------------------------------
def exercise4_keras_large():
    """
    Advanced CNN: deeper network with batch norm + dropout.
    Target ~85% test accuracy.
    """

    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images, test_images = train_images.astype("float32")/255, test_images.astype("float32")/255
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, stratify=train_labels, random_state=42
    )

    inputs = keras.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(train_images, train_labels,
              epochs=20, batch_size=64,
              validation_data=(val_images, val_labels),
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)])

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy (Advanced CNN): {test_acc:.3f}")

    preds = model.predict(test_images).argmax(axis=1)
    cm = confusion_matrix(test_labels, preds)
    np.savetxt("ex_keras/ex4_confusion_matrix.txt", cm, fmt="%d")


# -----------------------
# Run all exercises
# -----------------------
if __name__ == "__main__":
    exercise1_keras_basic()
    exercise2_keras_augmentation()
    exercise3_keras_pretrained()
    exercise4_keras_large()
