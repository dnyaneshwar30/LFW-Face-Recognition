import cv2
import os

def capture_face_data():
    dataset_dir = "dataset/my_face"
    os.makedirs(dataset_dir, exist_ok=True)

    print("Opening webcam...")
    print("Please look at the camera. Capturing 200 images for training.")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            count += 1
            # Save the captured face
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (224, 224)) # MobileNetV2 expects 224x224
            
            file_name_path = os.path.join(dataset_dir, f"user_{count}.jpg")
            cv2.imwrite(file_name_path, face_img)
            
            # Draw rectangle to show the user
            cv2.putText(frame, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
        cv2.imshow('Face Capture', frame)
        
        if cv2.waitKey(1) == 13 or count >= 200: # 13 is the Enter Key
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Data collection completed successfully!")

if __name__ == "__main__":
    capture_face_data()
