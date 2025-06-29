import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show fatal errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# ✅ Set upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Create the folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Load your trained model
model = tf.keras.models.load_model('efficientnet_checkpoint.keras', compile=False)

# ✅ Class labels (38 classes - from Kaggle Plant Village dataset)
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

# ✅ Preprocess function
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
            # Save image
            image_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_filename)

            # Preprocess & Predict
            img = preprocess_image(image_filename)
            pred = model.predict(img)
            predicted_class = int(np.argmax(pred))
            label = class_names[predicted_class]
            confidence = float(np.max(pred))

            prediction = label

    return render_template('index.html', prediction=prediction, confidence=confidence, image_filename=image_filename)

# ✅ Fix: ensure app runs when executed
if __name__ == '__main__':
    app.run(debug=True)
