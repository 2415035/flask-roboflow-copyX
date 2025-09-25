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
from datetime import date

# === Configuración Flask ===
app = Flask(__name__)

# === Configuración Roboflow ===
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="jBVSfkwNV6KBQ29SYJ5H"   # <<<<< tu API KEY de Roboflow
)
MODEL_ID = "pineapple-xooc7-5fxts/1"  # <<<<< tu modelo Roboflow

# === Configuración Supabase ===
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_API_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === Rutas principales ===
@app.route("/")
def home():
    return render_template("upload.html", title="Uploader")

@app.route("/dashboard")
def dashboard():
    response = supabase.table("predictions").select("*").execute()
    data = response.data if response.data else []

    if data:
        df = pd.DataFrame(data)
        # Asegúrate de que exista la columna 'clasevalidada' para graficar
        if 'clasevalidada' in df.columns:
            fig_bar = px.bar(df, x="clasevalidada", title="Conteo de clases validadas")
            fig_pie = px.pie(df, names="clasevalidada", title="Distribución de clases")
            graph_bar = fig_bar.to_html(full_html=False)
            graph_pie = fig_pie.to_html(full_html=False)
        else:
            graph_bar, graph_pie = None, None
    else:
        graph_bar, graph_pie = None, None

    return render_template("dashboard.html", graph_bar=graph_bar, graph_pie=graph_pie, data=data)

# === Procesar imagen con Roboflow + guardar en Supabase ===
@app.route('/process', methods=['POST'])
def process():
    try:
        image = request.files['imageFile']
        if not image:
            return jsonify({"error": "No se recibió la imagen"}), 400

        clase_validada_form = request.form.get("clasevalidada", "Palta")
        try:
            umbral_form = float(request.form.get("umbral", 0.5))
        except (TypeError, ValueError):
            umbral_form = 0.5

    image_file = request.files["imageFile"]
    if not image:
        return jsonify({"error": "No se envió ninguna imagen"}), 400
    image_bytes = image_file.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Guardar imagen temporalmente
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        result = CLIENT.infer(tmp.name, model_id=MODEL_ID)

    # Convertir imagen original
    img = Image.open(io.BytesIO(image_bytes))
    img = np.array(img)

    # Obtener clase válida y umbral desde el formulario
    clase_validada_form = request.form.get("clasevalidada", "Palta")
    
    try:
        umbral_form = float(request.form.get("umbral", 0.5))
    except (TypeError, ValueError):
        umbral_form = 0.5
    predicciones = result["predictions"]
    validos = sum(1 for p in predicciones if p["confidence"] >= umbral)
    invalidos = len(predicciones) - validos
    clase_validada = clase_validada_form

    # Dibujar resultados
    for pred in predicciones:
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        clase = pred["class"]
        conf = pred["confidence"]
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{clase} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Insertar en Supabase
    supabase.table("predictions").insert({
        "clasevalidada": clase_validada,
        "fecha": str(date.today()),
        "imagen": image_file.filename,
        "invalidos": invalidos,
        "validos": validos,
        "umbral": umbral,
        "predicciones": predicciones
    }).execute()

    # Convertir imagen a base64
    _, img_encoded = cv2.imencode(".png", img)
    img_base64 = base64.b64encode(img_encoded).decode("utf-8")

    return jsonify({
        "image": img_base64,
        "json": result
    })

# === Ejecutar local o en Render ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
