import io
import os
import base64
import tempfile
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, render_template, request, jsonify
from inference_sdk import InferenceHTTPClient
import plotly.express as px
from supabase import create_client, Client

# === Configuración Flask ===
app = Flask(__name__)

# === Configuración Roboflow ===
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="jBVSfkwNV6KBQ29SYJ5H"   # <<<<< pon tu API KEY de Roboflow
)
MODEL_ID = "pineapple-xooc7-5fxts/1"  # <<<<< reemplaza con el modelo que uses

# === Configuración Supabase ===
SUPABASE_URL = "https://ipgfsxnaohvsnnoxhdbv.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."  # tu key completa
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# === Rutas principales ===
@app.route("/")
def home():
    return render_template("upload.html", title="Uploader")

@app.route("/dashboard")
def dashboard():
    # Obtener datos desde Supabase
    response = supabase.table("predictions").select("*").execute()
    data = response.data if response.data else []

    if data:
        df = pd.DataFrame(data)

        # Gráfico de barras
        fig_bar = px.bar(df, x="label", title="Conteo de predicciones")
        graph_bar = fig_bar.to_html(full_html=False)

        # Gráfico de torta
        fig_pie = px.pie(df, names="label", title="Distribución de predicciones")
        graph_pie = fig_pie.to_html(full_html=False)
    else:
        graph_bar, graph_pie = None, None

    return render_template("dashboard.html", graph_bar=graph_bar, graph_pie=graph_pie, data=data)


# === Procesar imagen con Roboflow + guardar en Supabase ===
@app.route("/process", methods=["POST"])
def process():
    if "imageFile" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["imageFile"]
    image_bytes = image_file.read()

    # Guardar temporalmente
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        result = CLIENT.infer(tmp.name, model_id=MODEL_ID)

    # Cargar imagen original
    img = Image.open(io.BytesIO(image_bytes))
    img = np.array(img)

    # Dibujar predicciones en la imagen
    for pred in result["predictions"]:
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        clase = pred["class"]
        conf = pred["confidence"]

        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{clase} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # === Guardar cada predicción en Supabase ===
        supabase.table("predictions").insert({
            "filename": image_file.filename,
            "label": clase,
            "confidence": conf
        }).execute()

    # Convertir imagen a base64
    _, img_encoded = cv2.imencode(".png", img)
    img_base64 = base64.b64encode(img_encoded).decode("utf-8")

    return jsonify({
        "image": img_base64,
        "json": result
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
