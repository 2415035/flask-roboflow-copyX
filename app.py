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
    api_key=os.environ.get("ROBOFLOW_API_KEY")  # 🔒 Usa variable de entorno en Render
)
MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "pineapple-xooc7-5fxts/1")

# === Configuración Supabase ===
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_API_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === Rutas principales ===
@app.route("/")
def home():
    return render_template("dashboard.html", title="Panel de verificación")

@app.route("/dashboard")
def dashboard():
    response = supabase.table("predictions").select("*").execute()
    data = response.data if response.data else []

    graph_bar, graph_pie = None, None
    if data:
        df = pd.DataFrame(data)
        if "clasevalidada" in df.columns:
            fig_bar = px.bar(df, x="clasevalidada", title="Conteo de clases validadas")
            fig_pie = px.pie(df, names="clasevalidada", title="Distribución de clases")
            graph_bar = fig_bar.to_html(full_html=False)
            graph_pie = fig_pie.to_html(full_html=False)

    return render_template("dashboard.html", graph_bar=graph_bar, graph_pie=graph_pie, data=data)

# === Procesar imagen con Roboflow + guardar en Supabase ===
@app.route("/process", methods=["POST"])
def process():
    try:
        image_file = request.files.get("imageFile")
        if not image_file:
            return jsonify({"error": "No se recibió la imagen"}), 400

        image_bytes = image_file.read()

        # Guardar temporalmente para Roboflow
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            tmp.write(image_bytes)
            tmp.flush()
            result = CLIENT.infer(tmp.name, model_id=MODEL_ID)

        # Convertir imagen a numpy array
        img = Image.open(io.BytesIO(image_bytes))
        img = np.array(img)

        # Valores del formulario
        clase_validada = request.form.get("clasevalidada", "Palta")
        try:
            umbral = float(request.form.get("umbral", 0.5))
        except:
            umbral = 0.5

        # Contar predicciones
        predicciones = result.get("predictions", [])
        validos = sum(1 for p in predicciones if p["confidence"] >= umbral and p["class"] == clase_validada)
        invalidos = len(predicciones) - validos

        # Dibujar cajas
        for pred in predicciones:
            x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
            clase = pred["class"]
            conf = pred["confidence"]
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{clase} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Guardar en Supabase
        supabase.table("predictions").insert({
            "clasevalidada": clase_validada,
            "fecha": str(date.today()),
            "imagen": image_file.filename,
            "invalidos": invalidos,
            "validos": validos,
            "umbral": umbral,
            "predicciones": predicciones
        }).execute()

        # Imagen procesada a base64
        _, img_encoded = cv2.imencode(".png", img)
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")

        return jsonify({
            "image": img_base64,
            "json": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Run local o Render ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
