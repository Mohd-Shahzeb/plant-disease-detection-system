from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

# Load model
model = load_model("model/plant_model.h5")

# Load class labels
with open("model/classes.json", "r") as f:
    classes = json.load(f)

def prepare_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file.filename == '':
        return "No file uploaded"

    img = Image.open(file).convert('RGB')

    # Save image to static folder
    img_path = "static/uploaded.jpg"
    img.save(img_path)

    processed_img = prepare_image(img)

    prediction = model.predict(processed_img)

    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    # Simple disease info (you can expand)
    disease_info = {
        "Apple___healthy": "This plant is healthy. No action needed.",
        "Potato___Early_blight": "Fungal disease. Remove affected leaves and use fungicide.",
        "Potato___healthy": "Healthy potato plant.",
        "Tomato___Bacterial_spot": "Bacterial infection. Avoid overhead watering.",
        "Tomato___healthy": "Healthy tomato plant."
    }

    info = disease_info.get(predicted_class, "No info available")

    return render_template("result.html",
                           result=predicted_class,
                           confidence=round(confidence, 2),
                           image_path=img_path,
                           info=info)
if __name__ == "__main__":
    print("Starting Flask Server...")
    app.run(debug=True, use_reloader=False)