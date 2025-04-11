import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("best_model.keras")

# Emotion labels
emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Specific image path
image_path = r"C:\Users\Parth\OneDrive\Desktop\EmotionDetection\test\angry\PrivateTest_1221822.jpg"

# Load the image (already 48x48 grayscale)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"Error loading image: {image_path}")
    exit()

# Ensure the image is 48x48 pixels
if image.shape != (48, 48):
    print("Resizing image to 48x48...")
    image = cv2.resize(image, (48, 48))

# Normalize pixel values and reshape for model input
normalized_image = image.astype("float32") / 255.0
input_tensor = np.expand_dims(normalized_image, axis=(0, -1))  # Add batch and channel dimensions

# Predict emotion
predictions = model.predict(input_tensor)
emotion_label = emotions[np.argmax(predictions)]
print(f"Predicted Emotion: {emotion_label}")

# Display the predicted emotion
cv2.imshow("Emotion Detection", image)
cv2.putText(image, emotion_label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255), 1)
cv2.waitKey(0)
cv2.destroyAllWindows()
