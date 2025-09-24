from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import plotly.express as px
from supabase import create_client, Client

# === Configuración Flask ===
app = Flask(__name__)

# === Configuración Supabase ===
SUPABASE_URL = "https://ipgfsxnaohvsnnoxhdbv.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlwZ2ZzeG5hb2h2c25ub3hoZGJ2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg3Mzc4OTQsImV4cCI6MjA3NDMxMzg5NH0.Ia_CdNAiu5HKPsjc_J5e5Xu1Zoh-bwrBtnpvHP9-D_w"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === Rutas principales ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/dashboard")
def dashboard():
    # Obtener datos desde Supabase
    response = supabase.table("predictions").select("*").execute()
    data = response.data if response.data else []

    # Si hay datos, convertir a DataFrame
    if data:
        df = pd.DataFrame(data)

        # Gráfico de barras (conteo por label)
        fig_bar = px.bar(df, x="label", title="Conteo de predicciones")
        graph_bar = fig_bar.to_html(full_html=False)

        # Gráfico de torta (proporción de predicciones)
        fig_pie = px.pie(df, names="label", title="Distribución de predicciones")
        graph_pie = fig_pie.to_html(full_html=False)
    else:
        graph_bar, graph_pie = None, None

    return render_template("dashboard.html", graph_bar=graph_bar, graph_pie=graph_pie, data=data)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint para recibir predicciones y guardarlas en Supabase.
    Espera un JSON como:
    {
      "filename": "imagen1.jpg",
      "label": "Palta",
      "confidence": 0.92
    }
    """
    content = request.json
    filename = content.get("filename")
    label = content.get("label")
    confidence = content.get("confidence")

    # Insertar en Supabase
    supabase.table("predictions").insert({
        "filename": filename,
        "label": label,
        "confidence": confidence
    }).execute()

    return jsonify({"status": "success", "message": "Predicción guardada en Supabase"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
