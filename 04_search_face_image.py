import cv2
import sys
import os
import numpy as np

# Suppress TensorFlow logging to keep console clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

def process_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at '{image_path}'")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read the image. Please ensure it's a valid image file (like .jpg or .png).")
        return

    # Try to load the face lock model if it exists
    model_path = "face_lock_model.keras"
    model = None
    if os.path.exists(model_path):
        try:
            print("Loading Face Lock Model for recognition...")
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Could not load model: {e}")

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # detectMultiScale parameters: 
    # scaleFactor=1.1 compensates for faces closer/further to camera
    # minNeighbors=5 helps eliminate false positives
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print("No faces detected in this image.")
        return
        
    print(f"Detected {len(faces)} face(s). Extracting...")
    
    for i, (x, y, w, h) in enumerate(faces):
        # Extract (crop) the face region
        face_crop = img[y:y+h, x:x+w]
        
        label = "Detected Face"
        color = (255, 255, 0) # Cyan default color
        
        # Predict using the model if available
        if model is not None:
            try:
                resized_face = cv2.resize(face_crop, (224, 224))
                normalized_face = resized_face.astype("float32") / 255.0
                expanded_face = np.expand_dims(normalized_face, axis=0)
                
                prediction = model.predict(expanded_face, verbose=0)[0][0]
                
                if prediction >= 0.5:
                    label = f"User ({prediction*100:.1f}%)"
                    color = (0, 255, 0) # Green for User
                else:
                    label = f"Unknown ({(1 - prediction)*100:.1f}%)"
                    color = (0, 0, 255) # Red for Unknown
            except Exception as e:
                print("Prediction error:", e)
                
        # Draw a rectangle on the original image
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        # Put the label text above the face
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 1. Output: Show the cropped face in a separate window
        cv2.imshow(f'Output Face {i+1} - {label}', face_crop)
        
    # Resize the original image if it's too large to fit on screen
    max_dim = 800
    h, w = img.shape[:2]
    if h > max_dim or w > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
    # 2. Output: Show the original image with bounding boxes
    cv2.imshow('Image Search Result', img)
    print("Success! Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("\n=== Face Image Search & Extract ===")
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter the path to the input image file (e.g., C:\\path\\to\\image.jpg):\n> ")
        
    # Strip quotes in case the user dragged and dropped the file into the terminal
    image_path = image_path.strip('"\'')
    process_image(image_path)
