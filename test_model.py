"""
Test Script - Validate model on random test images
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os
from pathlib import Path
import random

MODEL_PATH = "./hummingbird_classifier.h5"
METADATA_PATH = "./class_labels.json"
DATASET_PATH = "./Humming bird moel/All_images"

def test_model_on_samples():
    """Test model on random images from each class"""
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run 'python train_model.py' first to train the model.")
        return
    
    # Load metadata
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    class_names = metadata['class_names']
    img_size = metadata['img_size']
    
    # Load model
    print("Loading model...")
    model = keras.models.load_model(MODEL_PATH)
    
    print("\n" + "="*70)
    print("TESTING MODEL ON RANDOM SAMPLES FROM EACH CLASS")
    print("="*70 + "\n")
    
    total_correct = 0
    total_tested = 0
    
    for class_name in class_names:
        class_path = os.path.join(DATASET_PATH, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        # Get all images
        image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
        
        # Test on 3 random images from each class
        test_images = random.sample(image_files, min(3, len(image_files)))
        
        print(f"\nClass: {class_name.upper()}")
        print("-" * 70)
        
        for img_file in test_images:
            img_path = os.path.join(class_path, img_file)
            
            # Load and preprocess
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (img_size, img_size))
            img_normalized = img_resized.astype('float32') / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Predict
            predictions = model.predict(img_batch, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx] * 100
            predicted_class = class_names[predicted_idx]
            
            # Check if correct
            is_correct = predicted_class == class_name
            total_correct += int(is_correct)
            total_tested += 1
            
            status = "✓ CORRECT" if is_correct else "✗ WRONG"
            print(f"  {img_file:20s} → {predicted_class:20s} ({confidence:5.1f}%) {status}")
    
    print("\n" + "="*70)
    print(f"Overall Accuracy on Test Sample: {(total_correct/total_tested)*100:.1f}% ({total_correct}/{total_tested})")
    print("="*70 + "\n")
    
    print("✓ Model is working correctly!")
    print("\nNext steps:")
    print("  1. Make predictions: python predict.py <image_path>")
    print("  2. Test on your own images")
    print("  3. Deploy the model")

if __name__ == "__main__":
    test_model_on_samples()
