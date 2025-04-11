from data_loader import create_generators
from model import build_emotion_cnn
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model():
    # Load data generators
    train_gen, val_gen, _ = create_generators()

    # Build the CNN model
    model = build_emotion_cnn()

    # Add callbacks for saving the best model and early stopping
    callbacks = [
        ModelCheckpoint("best_model.keras", save_best_only=True),  # Save best model in .keras format
        EarlyStopping(patience=5, restore_best_weights=True)       # Stop if no improvement
    ]

    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,               # Number of epochs
        callbacks=callbacks      # Use callbacks for optimization
    )

    return history

if __name__ == "__main__":
    train_model()
