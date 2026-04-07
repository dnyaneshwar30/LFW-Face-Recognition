# High-Accuracy Face Recognition with LFW Dataset

This repository contains a high-accuracy machine learning model for facial recognition using the Labeled Faces in the Wild (LFW) dataset. The original dataset and challenge details can be found on Kaggle.

## Overview

The model leverages the pre-trained `MobileNetV2` architecture, fine-tuning its top layers to extract powerful features for face classification. It uses deep learning and data augmentation to improve accuracy past the capabilities of simpler classification pipelines (such as PCA + SVM) and targets a higher robust accuracy rate.

### Features
- **Transfer Learning:** Built on `MobileNetV2` with ImageNet weights.
- **Data Augmentation:** Increases accuracy and prevents overfitting by dynamically generating varied images during training (random rotations, zooms, horizontal flips).
- **Automated Versioning:** Saves each run's model and logs metrics like accuracy, number of classes, and epochs.
- **Visual Plots:** Automatically plots training vs. validation accuracy for model evaluation.

## Requirements

This project is built to run easily in a Jupyter notebook environment such as Google Colab.

- **Kaggle Account:** To download the `jessicali9530/lfw-dataset`, you need your `kaggle.json` API token. 

## How to Run

1. Open `face_recognition.ipynb` in Google Colab or your local Jupyter environment.
2. Ensure you have your `kaggle.json` file ready. When prompted by the notebook, upload it to download the dataset.
3. Run all cells. The notebook will:
   - Configure the dataset.
   - Run the preprocessing and filtering logic.
   - Begin training the neural network.
   - Save the trained high-accuracy model as an `.h5` file in the `models/` folder.
   - Log the training process details in the `logs/` folder.
