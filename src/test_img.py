#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:57:15 2025

@author: jonathansetiawan
"""
import cv2
from keras.models import model_from_json
import numpy as np
import os

# Allowed file extensions
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg')

# Function to preprocess a single image
def preprocess_image(image_path):
    try:
        # Check file extension
        if not image_path.lower().endswith(ALLOWED_EXTENSIONS):
            raise ValueError(f"Unsupported file format: '{image_path}'. Allowed formats: {ALLOWED_EXTENSIONS}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image file '{image_path}' not found or invalid format.")
        
        # Resize and normalize the image
        resized_image = cv2.resize(img, (224, 224))
        normalized_image = resized_image / 255.0  # Normalize pixel values to [0, 1]
        return np.expand_dims(normalized_image, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Function to load the model and weights
def reload():
    try:
        with open('model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        
        loaded_model = model_from_json(loaded_model_json)
        
        # Load model weights
        weight_file = "ckpt.weights.h5"
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"Weight file '{weight_file}' not found.")
        
        loaded_model.load_weights(weight_file)
        print("Model and weights loaded successfully!")
        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to map predicted class to grade
def map_class_to_grade(predicted_class):
    # Mapping rules in directory names:
    # 0-2 -> Grade 1
    # 3-4 -> Grade 2
    # 5-6 -> Grade 3
    # 7-9 -> Grade 4
    
    grade_mapping = {
        1: range(0, 3),  # 0, 1, 2
        2: range(3, 5),  # 3, 4
        3: range(5, 7),  # 5, 6
        4: range(7, 10)  # 7, 8, 9
        }

    
    # Determine the grade based on the predicted class
    for grade, class_range in grade_mapping.items():
        if predicted_class in class_range:
            return grade
    
    return None

# Function to determine the true grade from the directory structure and crosscheck prediction
def crosscheck(image_path):
    try:
        # Extract the true class from the folder name
        true_class = int(os.path.basename(os.path.dirname(image_path)))
        
        # Map the true class to the corresponding grade
        true_grade = map_class_to_grade(true_class)
        return true_grade
    except ValueError:
        print("Error: Unable to determine true class from directory structure.")
        return None

# Function to predict and grade a single image, and check correctness
def predict_and_grade_image(image_path):
    # Preprocess the image
    print("Preprocessing the image...")
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is None:
        return

    # Reload the trained model
    print("Loading the model and weights...")
    loaded_model = reload()
    if loaded_model is None:
        return

    # Predict the image class
    print("Performing prediction...")
    prediction = loaded_model.predict(preprocessed_image).argmax(axis=1)[0]  # Get the predicted class index
    
    # Map the prediction to a grade
    predicted_grade = map_class_to_grade(prediction)
    
    # Determine the true grade
    true_grade = crosscheck(image_path)
    
    # Print results
    print(f"\n=== Prediction and Grading ===")
    print(f"Predicted Class Index: {prediction}")
    print(f"Predicted Grade: {predicted_grade}")
    print(f"True Grade (from directory): {true_grade}")
    
    # Check if the prediction is correct
    if predicted_grade == true_grade:
        print("Prediction is correct")
    else:
        print("Prediction is incorrect")
    
    return predicted_grade

# Run the prediction and grading for an individual image
if __name__ == "__main__":
    # Specify the path to the image you want to test
    image_path = './data/Testing_set/0/tomat (6).png'
    
    # Predict and grade the image
    predict_and_grade_image(image_path)
