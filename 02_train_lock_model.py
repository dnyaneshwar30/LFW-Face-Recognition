import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def prepare_unknown_faces(unknown_dir, num_faces_needed):
    """Downloads a few faces from sklearn LFW if the unknown directory is empty."""
    if not os.path.exists(unknown_dir):
        os.makedirs(unknown_dir)
        
    if len(os.listdir(unknown_dir)) < num_faces_needed:
        print("Downloading unknown faces for background class (this may take a moment)...")
        from sklearn.datasets import fetch_lfw_people
        try:
            # Get color images
            lfw = fetch_lfw_people(color=True)
            saved = 0
            for i, img in enumerate(lfw.images):
                if saved >= num_faces_needed:
                    break
                # sklearn images are float32 in [0, 1]. Convert to uint8 [0, 255]
                img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(unknown_dir, f"unknown_{i}.jpg"), img_bgr)
                saved += 1
            print(f"Saved {saved} unknown faces.")
        except Exception as e:
            print(f"Warning: Could not download LFW dataset: {e}")
            print("Please manually add some pictures of other people to 'dataset/unknown_faces'")

def train_model():
    user_dir = "dataset/my_face"
    unknown_dir = "dataset/unknown_faces"
    
    if not os.path.exists(user_dir) or len(os.listdir(user_dir)) == 0:
        print(f"Error: No user faces found in {user_dir}. Please run 01_capture_face.py first.")
        return
        
    num_user_faces = len(os.listdir(user_dir))
    
    # Prepare unknown faces (we want roughly double the unknown faces to prevent false positives)
    prepare_unknown_faces(unknown_dir, num_faces_needed=num_user_faces * 2)
    
    faces = []
    labels = []
    
    print("Loading image data...")
    # Load User Faces (Label 1)
    for img_name in os.listdir(user_dir):
        img_path = os.path.join(user_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            faces.append(img)
            labels.append(1) # 1 = User
            
    # Load Unknown Faces (Label 0)
    for img_name in os.listdir(unknown_dir):
        img_path = os.path.join(unknown_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            faces.append(img)
            labels.append(0) # 0 = Unknown
            
    # Convert to numpy arrays and normalize
    faces = np.array(faces, dtype="float32") / 255.0
    labels = np.array(labels)
    
    print(f"Total images: {len(faces)}")
    print(f"User faces: {np.sum(labels == 1)}, Unknown faces: {np.sum(labels == 0)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)
    
    print("Building MobileNetV2 model...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base model layers initially
    for layer in base_model.layers:
        layer.trainable = False
        
    # Unfreeze the top 20 layers for fine-tuning
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') # Binary classification: User or Not User
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    print("Training the model with Data Augmentation...")
    
    # Add Data Augmentation (this prevents the model from just memorizing your exact position)
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_test, y_test),
        epochs=15, # Increased epochs slightly since augmentation makes it harder
    )
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Final Validation Accuracy: {accuracy * 100:.2f}%")
    
    model_path = "face_lock_model.keras"
    model.save(model_path)
    print(f"Model saved successfully as {model_path}")

if __name__ == "__main__":
    train_model()
