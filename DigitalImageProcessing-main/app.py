from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from gradcam import generate_heatmap

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("lung_cancer_model.h5")

IMG_SIZE = 128

# Class labels (MUST match your dataset folders)
class_names = [
    "adenocarcinoma",
    "BenignCases",
    "large.cell.carcinoma",
    "MalignantCases",
    "normal",
    "squamous.cell.carcinoma"
]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    stage = None
    img_path = None
    heatmap_path = None

    if request.method == "POST":

        file = request.files["file"]

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Read image
            img = cv2.imread(filepath)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img_input = np.expand_dims(img, axis=0)

            # Prediction
            prediction = model.predict(img_input)
            class_index = np.argmax(prediction)
            predicted_class = class_names[class_index]

            print("Predicted:", predicted_class)

            # ==========================
            # USER-FRIENDLY OUTPUT
            # ==========================

            if predicted_class == "normal":
                result = "✅ Healthy (No Cancer Detected)"
                stage = "No risk"

            elif predicted_class == "BenignCases":
                result = "🟡 Non-Cancerous Tumor"
                stage = "Early stage (Not dangerous)"

            elif predicted_class == "MalignantCases":
                result = "🔴 Cancer Detected"
                stage = "Serious condition"

            elif predicted_class == "adenocarcinoma":
                result = "🔴 Lung Cancer"
                stage = "Adenocarcinoma (Common type)"

            elif predicted_class == "large.cell.carcinoma":
                result = "🔴 Lung Cancer"
                stage = "Large Cell (Fast growing)"

            elif predicted_class == "squamous.cell.carcinoma":
                result = "🔴 Lung Cancer"
                stage = "Squamous Cell (Smoking related)"

            else:
                result = "⚠️ Unable to classify"
                stage = "Try another image"

            # ==========================
            # HEATMAP (Tumor area)
            # ==========================
            try:
                heatmap = generate_heatmap(model, img_input)

                heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
                heatmap = np.uint8(255 * heatmap)

                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                superimposed = heatmap * 0.4 + img * 255
                heatmap_file = "heatmap_" + file.filename
                heatmap_path = os.path.join(UPLOAD_FOLDER, heatmap_file)

                cv2.imwrite(heatmap_path, superimposed)

            except Exception as e:
                print("Heatmap error:", e)

            img_path = filepath

    return render_template(
        "index.html",
        result=result,
        stage=stage,
        img_path=img_path,
        heatmap_path=heatmap_path
    )


if __name__ == "__main__":
    app.run(debug=True)