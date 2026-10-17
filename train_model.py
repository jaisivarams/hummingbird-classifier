"""
Hummingbird Species Classifier - Transfer Learning with MobileNetV2
Ready-to-run training script
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path
import json

# ============= CONFIGURATION =============
DATASET_PATH = "./Humming bird moel/All_images"
MODEL_SAVE_PATH = "./hummingbird_classifier.h5"
METADATA_SAVE_PATH = "./class_labels.json"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# ============= STEP 1: LOAD AND PREPROCESS DATA =============
print("="*60)
print("STEP 1: Loading and Preprocessing Data")
print("="*60)

def load_images_from_directory(data_path):
    """Load all images from folder structure"""
    images = []
    labels = []
    class_names = sorted(os.listdir(data_path))
    
    print(f"\nFound Classes: {class_names}\n")
    
    for label_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_path, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
        print(f"Loading {class_name}: {len(image_files)} images")
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                
                # Normalize (0-1)
                img = img.astype('float32') / 255.0
                
                images.append(img)
                labels.append(label_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    return np.array(images), np.array(labels), class_names

# Load data
X, y, class_names = load_images_from_directory(DATASET_PATH)
print(f"\nTotal images loaded: {len(X)}")
print(f"Image shape: {X[0].shape}")
print(f"Classes: {class_names}\n")

# Split data: 80% train+val, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
)

# Split train+val into train and val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=VALIDATION_SPLIT, random_state=42, stratify=y_temp
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}\n")

# Convert labels to one-hot encoding
num_classes = len(class_names)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# ============= STEP 2: BUILD MODEL =============
print("="*60)
print("STEP 2: Building Transfer Learning Model")
print("="*60)

# Load pre-trained MobileNetV2
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model weights (don't retrain)
base_model.trainable = False

# Build full model
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# ============= STEP 3: TRAIN MODEL =============
print("\n" + "="*60)
print("STEP 3: Training Model")
print("="*60 + "\n")

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    verbose=1
)

# ============= STEP 4: EVALUATE =============
print("\n" + "="*60)
print("STEP 4: Evaluating Model")
print("="*60 + "\n")

# Test accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%\n")

# Training accuracy
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
print(f"Train Loss: {train_loss:.4f}")
print(f"Train Accuracy: {train_accuracy*100:.2f}%\n")

# ============= STEP 5: SAVE MODEL =============
print("="*60)
print("STEP 5: Saving Model")
print("="*60 + "\n")

model.save(MODEL_SAVE_PATH)
print(f"✓ Model saved to: {MODEL_SAVE_PATH}")

# Save class labels
metadata = {
    'class_names': class_names,
    'img_size': IMG_SIZE,
    'test_accuracy': float(test_accuracy),
    'train_accuracy': float(train_accuracy)
}

with open(METADATA_SAVE_PATH, 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"✓ Class labels saved to: {METADATA_SAVE_PATH}")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nModel ready for predictions!")
print(f"Next: Run 'python predict.py <image_path>'")
