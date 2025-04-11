import os

# Suppress TensorFlow warnings and informational logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress info and warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations (optional)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

def build_emotion_cnn():
    model = Sequential([
        # Input layer
        Input(shape=(48, 48, 1)),  # Grayscale images of size 48x48

        # Convolutional Layer 1
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Convolutional Layer 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Convolutional Layer 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Flattening Layer
        Flatten(),

        # Fully Connected Layer
        Dense(256, activation='relu'),
        Dropout(0.5),  # Prevent overfitting

        # Output Layer (7 emotions)
        Dense(7, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Example usage:
if __name__ == "__main__":
    model = build_emotion_cnn()
    model.summary()  # Print model architecture
