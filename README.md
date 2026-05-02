# LFW Face Recognition & Personal Face Lock System

> *Last Updated: 2026-05-05 18:32:17 IST*

This repository contains two machine learning projects based on Face Recognition technologies:
1. **Multi-Class Celebrity Recognition (Original)** - A Jupyter Notebook demonstrating transfer learning on the Labeled Faces in the Wild (LFW) dataset.
2. **Personal Face Lock System (New!)** - A set of Python scripts that use your local webcam to capture your face, train a highly specialized binary classification model, and act as a secure, real-time "Face Lock" for your computer.

---

## Part 1: Celebrity Recognition (Original)
The `face_recognition.ipynb` notebook is designed to run in Google Colab. It uses `MobileNetV2` to classify and recognize various famous public figures from the LFW dataset. 

---

## Part 2: Personal Face Lock System
This feature allows you to build a personalized, real-time lock mechanism using your own webcam. 

> [!IMPORTANT]
> **Privacy First:** The `.gitignore` file in this repository is explicitly configured to prevent your personal photos (`dataset/`) and your trained model (`*.keras`) from being uploaded to GitHub. Your face data remains 100% private and secure on your local machine.

### Installation
To run the Face Lock system locally, you need to install the required Python dependencies:
```bash
pip install tensorflow opencv-python numpy scikit-learn
```

### How to Use

#### 1. Capture Your Face Data
Run the following script and look directly at your webcam. It will capture 200 pictures of you to build your personal dataset.
```bash
python 01_capture_face.py
```
*(Tip: Slowly move your head up, down, left, and right, and take your glasses off/on to help the AI learn your face better!)*

#### 2. Train the Model
Run the training script. This script automatically downloads "Unknown" faces from the LFW dataset to act as the negative class, and uses **Data Augmentation** to prevent overfitting. It then fine-tunes MobileNetV2 to distinguish you from everyone else.
```bash
python 02_train_lock_model.py
```

#### 3. Test the Live Face Lock
Once the model is trained, start the real-time webcam lock. It will display a green **"UNLOCKED"** box when it recognizes you, and a red **"LOCKED"** box if you step away or someone else steps in front of the camera.
```bash
python 03_face_lock.py
```

#### 4. (Bonus) Search a Photo
Want to test the AI on a static image instead of a webcam? Provide a path to an image file, and this script will extract the faces and classify them!
```bash
python 04_search_face_image.py
```
