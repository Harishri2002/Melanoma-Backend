import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load the model (ensure you adjust the path to where your model is stored)
model = tf.keras.models.load_model('Melanoma.h5')

# Define the class names (ensure the length matches the number of output classes in the model)
class_names = [
    'actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 
    'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion'
    # Add more class names here if needed
]

# Preprocess image (example: resize and normalize)
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))  # Resize to model input size
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive the image
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        img_file = request.files['image']

        # Preprocess the image
        img_array = preprocess_image(img_file.read())

        # Debugging input
        print("Input shape:", img_array.shape)

        # Modify the model architecture here for input compatibility
        # Apply GlobalAveragePooling2D to reduce dimensions
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(256, (11, 11), activation='relu', padding='valid'),
            tf.keras.layers.GlobalAveragePooling2D(),  # Reduce the dimensionality
            tf.keras.layers.Dense(9, activation='softmax')  # Number of classes (9)
        ])

        # Compile the model (adjust optimizer, loss, and metrics)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Predict using the model
        predictions = model.predict(img_array)

        # Debugging predictions
        print("Predictions shape:", predictions.shape)
        print("Predictions content:", predictions)

        # Validate predictions shape (expecting a 2D array with shape (1, num_classes))
        if len(predictions.shape) != 2 or predictions.shape[1] != len(class_names):
            raise ValueError(f"Unexpected predictions shape: {predictions.shape}")

        # Get the predicted class and confidence
        predicted_class_index = np.argmax(predictions[0])  # Get the index of the highest probability
        confidence = float(predictions[0][predicted_class_index])  # Get the confidence score for that class

        # Check if the predicted class is melanoma (index 3)
        melanoma_confidence = float(predictions[0][3])  # Get the melanoma confidence (class index 3)

        # Ensure the predicted class index is within bounds
        if predicted_class_index >= len(class_names):
            raise ValueError(f"Predicted class index {predicted_class_index} out of bounds for class names list.")

        # Get the predicted class name
        predicted_class_name = class_names[predicted_class_index]  # Safely access the class name

        # Prepare response data
        response_data = {
            'class': predicted_class_name,
            'confidence': float(confidence) * 100,  # Convert to percentage and ensure it's a float
            'melanoma_confidence': float(melanoma_confidence) * 100  # Convert melanoma confidence to float percentage
        }


        # If the predicted class is not melanoma, include the highest confidence class
        if predicted_class_index != 3:  # If it's not melanoma
            # Find the class with the highest confidence other than melanoma
            non_melanoma_classes = [(class_names[i], float(predictions[0][i]) * 100) for i in range(len(class_names)) if i != 3]
            non_melanoma_classes.sort(key=lambda x: x[1], reverse=True)
            response_data['alternative_class'] = non_melanoma_classes[0]

        # Return the result
        return jsonify(response_data)

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
