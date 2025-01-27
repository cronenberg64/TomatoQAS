#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 01:22:45 2024
@author: jonathansetiawan
"""

import numpy as np
import cv2
import os

# Specify the image path
image_path = './data/Training_set/0/001.jpg'

# Check if the file exists
if not os.path.exists(image_path):
    print(f"Error: The file at {image_path} does not exist.")
else:
    print("File exists. Proceeding to load the image.")

    # Load the image
    img = cv2.imread(image_path)

    # Validate if the image was loaded successfully
    if img is None:
        print("Error: Failed to load the image. It might be corrupted or an unsupported format.")
    else:
        print("Image loaded successfully.")

        # Ensure the image is 3-channel (convert if grayscale)
        if len(img.shape) == 2:
            print("Image is grayscale. Converting to 3-channel format...")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Apply denoising with less aggressive parameters
        denoising_img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
        print("Denoising applied successfully with reduced sensitivity.")

        # Resize the image for faster processing while preserving quality
        resized_img = cv2.resize(denoising_img, (1920, 1080))
        print("Image resized to 1000x600 for processing.")

        # Initialize GrabCut mask and models
        mask = np.zeros(resized_img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Define a larger region of interest for GrabCut
        rect = (30, 30, resized_img.shape[1], resized_img.shape[0])
        print(f"Region of Interest for GrabCut: {rect}")

        # Apply GrabCut algorithm with iterative fine-tuning
        cv2.grabCut(resized_img, mask, rect, bgdModel, fgdModel, 15, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Apply morphological operations for post-processing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        print("Segmentation mask refined with morphological operations.")

        # Use contour detection to filter the largest contour
        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # Create a new mask for the largest contour
            refined_mask = np.zeros_like(mask2)
            cv2.drawContours(refined_mask, [largest_contour], -1, (1), thickness=cv2.FILLED)

            # Apply refined mask to the image
            segmented_img = resized_img * refined_mask[:, :, np.newaxis]
            print("Contour-based refinement applied.")
        else:
            print("No contours detected. Using initial GrabCut mask.")
            segmented_img = resized_img * mask2[:, :, np.newaxis]

        # Save and display the results
        output_path = 'segmented_output_refined.jpg'
        cv2.imwrite(output_path, segmented_img)
        print(f"Segmented image saved to {output_path}")

        # Display the original and segmented images
        cv2.imshow("Original Image", resized_img)
        cv2.imshow("Segmented Image (Refined)", segmented_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
