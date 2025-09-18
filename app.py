import io
from PIL import Image
from roboflow import Roboflow
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import tempfile
import base64

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('upload.html', title='Equisd')

@app.route("/dashboard")
def dashboard():
    return render_template('dashboard.html', title='Equisd')

@app.route("/process", methods=['POST'])
def process():
    if 'imageFile' not in request.files:
        return "No image provided", 400
    
    image_file = request.files['imageFile']
    image_bytes = image_file.read()

    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp.write(image_bytes)
        tmp.flush()

        # 🔹 Usamos Roboflow SDK
        rf = Roboflow(api_key="jBVSfkwNV6KBQ29SYJ5H")
        project = rf.workspace().project("pineapple-xooc7-5fxts")
        model = project.version(1).model
        result = model.predict(tmp.name).json()

    # Cargar la imagen original
    img = Image.open(io.BytesIO(image_bytes))
    img = np.array(img)

    for pred in result["predictions"]:
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        clase = pred["class"]
        conf = pred["confidence"]

        x1, y1 = x - w//2, y - h//2
        x2, y2 = x + w//2, y + h//2

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(
            img,
            f"{clase} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
    
    _, img_encoded = cv2.imencode(".png", img)
    img_base64 = base64.b64encode(img_encoded).decode("utf-8")

    return jsonify({
        "image": img_base64,
        "json": result
    })

if __name__ == "__main__":
    app.run(debug=True)
