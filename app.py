import os

# Set TensorFlow environment variables before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model (should already be downloaded during build)
MODEL_PATH = "efficientnet_checkpoint.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found! Make sure it was downloaded during build.")

print("🤖 Loading EfficientNet model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")

class_names = [
    "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
    "Blueberry Healthy", "Cherry Powdery Mildew", "Cherry Healthy",
    "Corn Cercospora", "Corn Common Rust", "Corn Northern Leaf Blight", "Corn Healthy",
    "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", "Grape Healthy",
    "Orange Huanglongbing", "Peach Bacterial Spot", "Peach Healthy",
    "Pepper Bell Bacterial Spot", "Pepper Bell Healthy",
    "Potato Early Blight", "Potato Late Blight", "Potato Healthy",
    "Raspberry Healthy", "Soybean Healthy", "Squash Powdery Mildew",
    "Strawberry Leaf Scorch", "Strawberry Healthy",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight", "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot", "Tomato Spider Mites", "Tomato Target Spot",
    "Tomato Mosaic Virus", "Tomato Yellow Leaf Curl Virus", "Tomato Healthy"
]

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_filename = None

    if request.method == 'POST':
        file = request.files['image']
        if file and file.filename:
            image_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_filename)

            try:
                img = preprocess_image(image_filename)
                pred = model.predict(img)
                predicted_class = int(np.argmax(pred))
                label = class_names[predicted_class]
                confidence = float(np.max(pred))
                prediction = label
            except Exception as e:
                print(f"Error during prediction: {e}")
                prediction = "Error processing image"
                confidence = 0.0

    return render_template('index.html', 
                         prediction=prediction, 
                         confidence=confidence, 
                         image_filename=image_filename)

if __name__ == '__main__':
    # For local development
    app.run(host='0.0.0.0', port=5000, debug=True)
else:
    # For production (Render will use gunicorn)
    pass
