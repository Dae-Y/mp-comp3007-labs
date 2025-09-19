import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure consistent Keras import
from tensorflow.keras import layers

def exercise1_keras_basic():
    """
    Description: Basic NN using Keras to classify CIFAR10 images
    """
    print("--- Running Exercise 1: Basic Convnet ---")
    
    # Load CIFAR-10 data
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Preprocess data (normalizing)
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255

    # Split the training set into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, stratify=train_labels, random_state=42)

    # Construct a model with 3 conv layers and two max pooling layers in between
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model using RMSprop optimizer
    print("Exercise 1 - compile model using RMSprop optimizer")
    model.compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Train the model
    model.fit(train_images, train_labels,
              epochs=5,
              batch_size=32,
              validation_data=(val_images, val_labels))

    # Evaluate the model on test data
    test_loss_rms, test_acc_rms = model.evaluate(test_images, test_labels)
    print(f"Test accuracy (RMSprop): {test_acc_rms:.3f}")

    # Predict the labels of the training images
    train_predictions_rms = model.predict(train_images)
    train_pred_labels_rms = train_predictions_rms.argmax(axis=1)
    
    # Compute the confusion matrix
    cm_rms = confusion_matrix(train_labels, train_pred_labels_rms)

    # Compile the model using Adam optimizer
    print("Exercise 1 - compile model using Adam optimizer")
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Train the model
    model.fit(train_images, train_labels,
              epochs=5,
              batch_size=32,
              validation_data=(val_images, val_labels))

    # Evaluate the model on test data
    test_loss_adam, test_acc_adam = model.evaluate(test_images, test_labels)
    print(f"Test accuracy (Adam): {test_acc_adam:.3f}")

    # Predict the labels of the training images
    train_predictions_adam = model.predict(train_images)
    train_pred_labels_adam = train_predictions_adam.argmax(axis=1)

    # Compute the confusion matrix
    cm_adam = confusion_matrix(train_labels, train_pred_labels_adam)

    # Plot the confusion matrix
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_rms, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix of Training Samples (RMS)")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_adam, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix of Training Samples (Adam)")
    
    # plt.savefig('ex1/confusion_matrix.png') # Uncomment to save the figure
    plt.show()

def exercise2_keras_augmentation():
    """
    Description: Basic NN using Keras to classify CIFAR10 images
                 with data augmentation and dropout
    """
    print("\n--- Running Exercise 2: Convnet with Augmentation ---")

    # Load CIFAR-10 data
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Preprocess data (normalizing)
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255

    # Split the training set into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, stratify=train_labels, random_state=42)

    # Data augmentation layer
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    # Rebuild the model with augmentation, dropout, and batch normalisation
    inputs = keras.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model using adam optimizer
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Train the model
    model.fit(train_images, train_labels,
              epochs=20,
              batch_size=32, # Using a larger batch size is common here
              validation_data=(val_images, val_labels))

    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc:.3f}")

    # Predict the labels of the test images and compute confusion matrix
    print("\nPredicting labels and computing confusion matrix...")
    test_predictions = model.predict(test_images)
    test_pred_labels = np.argmax(test_predictions, axis=1)
    cm = confusion_matrix(test_labels, test_pred_labels)
    print("Confusion Matrix:\n", cm)

def exercise3_keras_pretrained():
    """
    Description: Fine-tuning a pre-trained convnet to classify CIFAR10 images
    """
    print("\n--- Running Exercise 3: Fine-Tuning a Pre-trained VGG16 ---")

    # Load CIFAR-10 data
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Preprocess data but DO NOT normalize yet, as VGG16 has its own preprocess function
    train_images = train_images.astype("float32")
    test_images = test_images.astype("float32")
    
    # Split the training set into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, stratify=train_labels, random_state=42)

    # Load VGG16 model with pretrained ImageNet weights, excluding the top layers
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Freeze the entire convolutional base
    conv_base.trainable = False

    # Rebuild model with pre-trained VGG16
    inputs = keras.Input(shape=(32, 32, 3))
    x = keras.applications.vgg16.preprocess_input(inputs) # VGG16-specific preprocessing
    x = conv_base(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    # Compile the model with a smaller learning rate for fine-tuning
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Train the model
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="fine_tuning.keras",
            save_best_only=True,
            monitor="val_loss")
    ]
    
    history = model.fit(train_images, train_labels,
                        epochs=10,
                        batch_size=16,
                        validation_data=(val_images, val_labels),
                        callbacks=callbacks)

    # Evaluate the model on test data
    model = keras.models.load_model("fine_tuning.keras")
    test_loss_convnet, test_acc_convnet = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc_convnet:.3f}")

def exercise4_keras_large():
    """
    Description: A larger network with more layers, batch norm and drop-out
                 trained from scratch. Should achieve around 85% accuracy.
    """
    print("\n--- Running Exercise 4: Larger Custom Convnet ---")
    
    # Load CIFAR-10 data
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Preprocess data (normalizing)
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255

    # Split the training set into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, stratify=train_labels, random_state=42)

    # Construct an improved model
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x) # This was the missing line

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # COMPILE STEP WAS MISSING
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Train the model with a greater number of epochs and early stopping
    model.fit(train_images, train_labels,
              epochs=50,  # Increased epochs for a larger model
              batch_size=64,
              validation_data=(val_images, val_labels),
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])

    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc:.3f}")

    # Predict the labels of the test images and compute confusion matrix
    print("\nPredicting labels and computing confusion matrix...")
    test_predictions = model.predict(test_images)
    test_pred_labels = np.argmax(test_predictions, axis=1)
    cm = confusion_matrix(test_labels, test_pred_labels)
    print("Confusion Matrix:")
    print(cm)

# To run the code, call the functions you want to execute
if __name__ == '__main__':
    exercise1_keras_basic()
    exercise2_keras_augmentation()
    exercise3_keras_pretrained()
    exercise4_keras_large()