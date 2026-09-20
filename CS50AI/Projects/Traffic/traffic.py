import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Load image data and corresponding labels
    images, labels = load_data(sys.argv[1])

    # Convert labels to one-hot encoding
    labels = tf.keras.utils.to_categorical(labels)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file if output filename is provided
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory data_dir.

    Assume data_dir has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Within each category directory, read image
    files, resize each image to (IMG_WIDTH, IMG_HEIGHT), and scale pixels.

    Return tuple (images, labels). images is a list of all of the
    images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions (IMG_WIDTH, IMG_HEIGHT, 3). labels is
    a list of integer labels, representing the category for each image.
    """
    images = []
    labels = []

    for cat in range(NUM_CATEGORIES):
        cat_path = os.path.join(data_dir, str(cat))

        if not os.path.isdir(cat_path):
            continue

        for file_name in os.listdir(cat_path):
            file_path = os.path.join(cat_path, file_name)

            # Read image using OpenCV
            img = cv2.imread(file_path)

            if img is not None:
                # Resize image to targeted dimensions
                resized_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

                # Scale pixel values to range [0, 1]
                normalized_img = resized_img / 255.0

                images.append(normalized_img)
                labels.append(cat)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume it takes
    input_shape (IMG_WIDTH, IMG_HEIGHT, 3).
    The output layer should have NUM_CATEGORIES units and a softmax activation.
    """
    model = tf.keras.models.Sequential([
        # First Convolutional Layer
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Second Convolutional Layer
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units into 1D vector
        tf.keras.layers.Flatten(),

        # Hidden dense layer with dropout regularization
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Output layer with softmax activation for category classification
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile model with cross-entropy loss and Adam optimizer
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "_main_":
    main()
