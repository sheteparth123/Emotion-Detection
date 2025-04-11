import os

def check_dataset_structure():
    train_path = "train"
    test_path = "test"
    
    # Check if train and test folders exist
    if not os.path.exists(train_path):
        print(f"ERROR: '{train_path}' folder is missing!")
        return
    if not os.path.exists(test_path):
        print(f"ERROR: '{test_path}' folder is missing!")
        return
    
    # Count images in each emotion folder
    print("\nTraining Data:")
    for emotion in os.listdir(train_path):
        emotion_path = os.path.join(train_path, emotion)
        if os.path.isdir(emotion_path):
            num_images = len(os.listdir(emotion_path))
            print(f"{emotion}: {num_images} images")
    
    print("\nTesting Data:")
    for emotion in os.listdir(test_path):
        emotion_path = os.path.join(test_path, emotion)
        if os.path.isdir(emotion_path):
            num_images = len(os.listdir(emotion_path))
            print(f"{emotion}: {num_images} images")

if __name__ == "__main__":
    check_dataset_structure()
