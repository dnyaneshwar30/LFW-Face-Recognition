import cv2
import numpy as np
import tensorflow as tf
import os

def run_face_lock():
    model_path = "face_lock_model.keras"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run 02_train_lock_model.py first.")
        return

    print("Loading Face Lock Model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("Starting Face Lock System. Press 'q' or 'Enter' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Default state is locked
        lock_status = "LOCKED"
        color = (0, 0, 255) # Red for locked
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            
            try:
                # Preprocess for the model
                resized_face = cv2.resize(face_img, (224, 224))
                normalized_face = resized_face.astype("float32") / 255.0
                expanded_face = np.expand_dims(normalized_face, axis=0) # Add batch dimension
                
                # Predict
                prediction = model.predict(expanded_face, verbose=0)[0][0]
                
                # Confidence threshold (0.5 is the standard for binary classification)
                if prediction >= 0.5:
                    lock_status = "UNLOCKED"
                    color = (0, 255, 0) # Green for unlocked
                    label = f"User ({prediction*100:.1f}%)"
                else:
                    label = f"Unknown ({(1 - prediction)*100:.1f}%)"
                    
                # Draw bounding box around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            except Exception as e:
                print("Prediction error:", e)
                
        # Display the lock status on the main screen
        cv2.putText(frame, lock_status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
        cv2.imshow('Face Lock System', frame)
        
        key = cv2.waitKey(1)
        if key == 13 or key == ord('q'): # 13 is Enter Key
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_lock()
