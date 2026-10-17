import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json

def predict_bird_species(image_path):
    """Use trained model to predict hummingbird species"""
    
    # Load model
    print("Loading model...")
    model = keras.models.load_model('hummingbird_classifier.h5')
    
    # Load metadata (class names)
    with open('class_labels.json', 'r') as f:
        metadata = json.load(f)
    
    class_names = metadata['class_names']
    img_size = metadata['img_size']
    
    # Load image
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not load image!")
        return None
    
    # Preprocess image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_normalized = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Make prediction
    print("Making prediction...")
    predictions = model.predict(img_batch, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100
    predicted_class = class_names[predicted_idx]
    
    # Print results
    print("\n" + "="*60)
    print(f"Image: {image_path}")
    print("="*60)
    print(f"\n✓ PREDICTED: {predicted_class.upper()}")
    print(f"✓ Confidence: {confidence:.2f}%\n")
    
    print("All Species Probabilities:")
    for idx, class_name in enumerate(class_names):
        prob = predictions[0][idx] * 100
        bar = "█" * int(prob / 5)  # Visual bar
        print(f"  {class_name:20s} {prob:6.2f}% {bar}")
    
    print("\n" + "="*60 + "\n")
    
    return predicted_class, confidence

# Run prediction

if __name__ == "__main__":
    print("🐦 Hummingbird Species Classifier")
    print("=" * 60)
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        image_path = input("Enter image path: ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("\n✓ Thanks for using the classifier! Goodbye! 👋")
            break
        
        if not image_path:
            print("Please enter a valid path.\n")
            continue
        
        result = predict_bird_species(image_path)
        
        if result:
            species, confidence = result
            print(f"✓ Successfully identified: {species} ({confidence:.1f}%)\n")