import io
from PIL import Image
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import tempfile

app = Flask(__name__)
import base64

@app.route("/")
def home():
    return render_template('upload.html', title='Equisd')

@app.route("/dashboard")
def dashboard():
    return render_template('dashboard.html', title='Equisd')

@app.route("/process", methods=['POST'])
def process():
    # include code to process the image using sdk inference of roboflow


    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="jBVSfkwNV6KBQ29SYJ5H" # <<<< Cambiar
    )

    # Get image from the request
    if 'imageFile' not in request.files:
        return "No image provided", 400
    
    image_file = request.files['imageFile']
    image_bytes = image_file.read()

    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        result = CLIENT.infer(tmp.name, model_id="pineapple-xooc7-5fxts/1") # <<<< Cambiar

    # Cargar la imagen original
    img = Image.open(io.BytesIO(image_bytes))
    img = np.array(img)
    for pred in result["predictions"]:
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        clase = pred["class"]
        conf = pred["confidence"]

        # calcular esquinas del bounding box
        x1, y1 = x - w//2, y - h//2
        x2, y2 = x + w//2, y + h//2

        # dibujar rectángulo en negro
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)

        # dibujar texto en negro
        cv2.putText(
            img,
            f"{clase} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),  # color negro
            2
        )
    
    # Convert the image to base64
    _, img_encoded = cv2.imencode(".png", img)
    img_base64 = base64.b64encode(img_encoded).decode("utf-8")

    return jsonify({
        "image": img_base64,
        "json": result
    })
    
if __name__ == "__main__":
    app.run(debug=True)
