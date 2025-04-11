from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(batch_size=64):
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,          # Normalize pixel values (0-255 -> 0-1)
        validation_split=0.2,    # Split training data into train/validation sets
        rotation_range=15,       # Random rotation
        width_shift_range=0.1,   # Random horizontal shift
        height_shift_range=0.1,  # Random vertical shift
        shear_range=0.1,         # Shear transformation
        zoom_range=0.2,          # Random zoom
        horizontal_flip=True     # Random horizontal flip
    )

    # No augmentation for testing
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Training data generator
    train_gen = train_datagen.flow_from_directory(
        "train",                 # Path to training folder
        target_size=(48, 48),    # Resize images to 48x48 pixels (FER standard)
        color_mode="grayscale",  # Convert images to grayscale
        class_mode="categorical",# Use categorical labels (one-hot encoding)
        batch_size=batch_size,
        subset="training"        # Use training subset (80%)
    )

    # Validation data generator
    val_gen = train_datagen.flow_from_directory(
        "train",
        target_size=(48, 48),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=batch_size,
        subset="validation"      # Use validation subset (20%)
    )

    # Testing data generator
    test_gen = test_datagen.flow_from_directory(
        "test",
        target_size=(48, 48),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False            # Do not shuffle for testing (to match labels)
    )

    return train_gen, val_gen, test_gen

# Example usage:
if __name__ == "__main__":
    train_gen, val_gen, test_gen = create_generators()
