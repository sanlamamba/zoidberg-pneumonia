from flask import Flask, request, jsonify
import joblib
import os
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load the model globally
model = None

def load_model():
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'decision_classifier.pkl')
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f'Error loading model: {e}')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def preprocess_image(image_file):
    """Convert image to a format suitable for prediction."""
    try:
        # Open the image and convert it to RGB
        image = Image.open(image_file.stream).convert('RGB')
        # Resize image to the size your model expects
        image = image.resize((128, 128))  # Example size, adjust as needed
        image = np.array(image)
        image = image / 255.0  # Normalize the image
        return image[np.newaxis, ...]  # Add batch dimension
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def predict_from_image(model, preprocessed_image):
    if preprocessed_image is not None:
        print(":sparkles: Predicting image :sparkles:")
        prediction = model.predict(preprocessed_image)
        return prediction
    else:
        return "Error processing image"

@app.route('/api/image', methods=['POST'])
def image():
    print(":camera: Image endpoint called", request.files)
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400
    if image_file and allowed_file(image_file.filename):
        preprocessed_image = preprocess_image(image_file)
        if model is not None and preprocessed_image is not None:
            prediction = predict_from_image(model, preprocessed_image)
            return jsonify({'prediction': str(prediction)})
        else:
            return jsonify({'error': 'Failed to process image or model not loaded'}), 500
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

@app.route('/api')
def home():
    return jsonify({'message': 'Hello World!'})

if __name__ == '__main__':
    load_model()
    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, port=port)
