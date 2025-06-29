import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show fatal errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

app = Flask(__name__)

# ✅ Set upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Google Drive file ID of your model
MODEL_FILE_ID = 'YOUR_FILE_ID_HERE'  # <-- Replace with your Drive file ID
MODEL_PATH = 'efficientnet_checkpoint.keras'

# ✅ Download model from Google Drive if not present
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# ✅ Load the model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ✅ Class labels (38 classes from PlantVillage dataset)
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

# ✅ Image preprocessing
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ✅ Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_filename = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            image_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_filename)

            img = preprocess_image(image_filename)
            pred = model.predict(img)
            predicted_class = int(np.argmax(pred))
            label = class_names[predicted_class]
            confidence = float(np.max(pred))
            prediction = label

    return render_template('index.html', prediction=prediction, confidence=confidence, image_filename=image_filename)

# ✅ Required to run on Render
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
