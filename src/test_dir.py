#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 00:01:48 2024

@author: jonathansetiawan
"""
import cv2
from keras.models import model_from_json
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os
import glob

# Path to Testing Data
path_image = './data/Validation_set'

# Function to Map Classes to Grades
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

    
    for grade, class_range in grade_mapping.items():
        if predicted_class in class_range:
            return grade
    return -1  # Default fallback for unexpected values

# Function to Read and Preprocess Testing Data
def read_data(path_image):
    array = []
    label = []
    folder = os.listdir(path_image)
    
    for j, out_folder in enumerate(folder):
        image_path = os.path.join(path_image, out_folder)
        
        # Handle multiple image formats
        image_list = glob.glob(image_path + '/*.jpg') + \
                     glob.glob(image_path + '/*.jpeg') + \
                     glob.glob(image_path + '/*.png')
        
        for i, image in enumerate(image_list):
            img = cv2.imread(image)
            resized_image = cv2.resize(img, (224, 224))
            array.append(resized_image / 255.0)  # Normalize the image
            label.append(int(j))  # Assign a numeric label based on folder index
    
    # Convert to numpy arrays
    x_test = np.asarray(array, dtype='float32')
    y_test = np.asarray(label, dtype='int')
    
    # Map actual classes to grades
    yy_test = [map_class_to_grade(cls) for cls in y_test]
    
    # Filter out invalid grades (-1)
    valid_indices = [i for i, grade in enumerate(yy_test) if grade != -1]
    x_test = x_test[valid_indices]
    yy_test = np.array(yy_test)[valid_indices]
    
    return x_test, yy_test

# Function to Load the Model and Weights
def reload():
    print("Current Working Directory:", os.getcwd())
    try:
        with open('model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        
        loaded_model = model_from_json(loaded_model_json)
        weight_file = "ckpt.weights.h5"
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"Weight file '{weight_file}' not found.")
        
        loaded_model.load_weights(weight_file)
        print("Model and weights loaded successfully!")
        loaded_model.summary()
        
        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

# Function to Evaluate Model on Testing Data
def score():
    print("Reading testing data...")
    X_test, YY_test = read_data(path_image)
    if len(X_test) == 0 or len(YY_test) == 0:
        print("No valid testing data found. Please check the Testing_set directory.")
        return
    
    print("Testing data loaded successfully!")
    print("Loading the model and weights...")
    loaded_model = reload()
    
    print("Performing predictions...")
    Y_pred = loaded_model.predict(X_test).argmax(axis=1)  # Get predicted class indices
    Y_pred_grades = np.array([map_class_to_grade(pred) for pred in Y_pred])  
    
    # Filter out invalid predictions (-1)
    valid_indices = [i for i, grade in enumerate(Y_pred_grades) if grade != -1]
    Y_pred_grades = Y_pred_grades[valid_indices]
    YY_test = YY_test[valid_indices]
    
    if len(YY_test) == 0 or len(Y_pred_grades) == 0:
        print("No valid predictions to evaluate. Please check the testing data or model.")
        return
    
    # Calculate and display the accuracy
    score = accuracy_score(YY_test, Y_pred_grades)
    print("\n=== Model Performance ===")
    print("Accuracy:", score)
    
    # Display classification report
    print("\nClassification Report (Grades):\n", classification_report(YY_test, Y_pred_grades))

# Run the Evaluation
if __name__ == "__main__":
    score()
