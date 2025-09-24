from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
import plotly.graph_objs as go
import base64

app = Flask(__name__)

# === Configuración de SUPABASE ===
SUPABASE_URL = "https://ipgfsxnaohvsnnoxhdbv.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."  # tu API KEY
SUPABASE_TABLE = "predictions"

HEADERS = {
    "apikey": SUPABASE_API_KEY,
    "Authorization": f"Bearer {SUPABASE_API_KEY}"
}

# ==========================
# RUTA PRINCIPAL
# ==========================
@app.route("/")
def home():
    return render_template('upload.html', title='Subir imagen')


# ==========================
# RUTA DE PROCESAMIENTO (simulada)
# ==========================
@app.route("/process", methods=["POST"])
def process():
    """
    Aquí normalmente iría tu lógica de ML/Roboflow para procesar la imagen.
    Por ahora devolveré algo simulado para que el flujo funcione.
    """

    file = request.files.get("imageFile")
    if not file:
        return jsonify({"error": "No se envió archivo"}), 400

    # Convertir la imagen subida a base64 para mostrar en frontend
    img_bytes = file.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # Ejemplo de JSON simulado de predicciones
    result_json = {
        "predictions": [
            {"class": "ripen", "confidence": 0.92},
            {"class": "unripen", "confidence": 0.44}
        ]
    }

    return jsonify({
        "image": img_b64,
        "json": result_json
    })


# ==========================
# RUTA PARA GUARDAR EN SUPABASE
# ==========================
@app.route("/save_prediction", methods=["POST"])
def save_prediction():
    try:
        data = request.json

        url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
        response = requests.post(
            url,
            headers={**HEADERS, "Content-Type": "application/json"},
            json=data
        )

        if response.status_code in [200, 201]:
            return jsonify({"status": "ok", "data": response.json()})
        else:
            return jsonify({"status": "error", "msg": response.text}), 400
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500


# ==========================
# DASHBOARD CON GRÁFICOS
# ==========================
@app.route("/dashboard")
def dashboard():
    try:
        url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}?select=*"
        response = requests.get(url, headers=HEADERS)
        data = response.json()

        if not data:
            return "No hay datos disponibles desde Supabase."

        df = pd.DataFrame(data)

        if df.empty:
            return "No hay datos para mostrar."

        # Convertir columna fecha si existe
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date

        # ========== Gráfico de torta ==========
        if "claseValidada" in df.columns:
            pie_data = df["claseValidada"].value_counts().reset_index()
            pie_chart = go.Figure(data=[go.Pie(labels=pie_data['index'], values=pie_data['claseValidada'])])
            pie_html = pie_chart.to_html(full_html=False)
        else:
            pie_html = "<p>No hay columna 'claseValidada'.</p>"

        # ========== Gráfico de barras ==========
        if "fecha" in df.columns:
            bar_data = df.groupby("fecha").size().reset_index(name="total")
            bar_chart = go.Figure(data=[go.Bar(x=bar_data["fecha"], y=bar_data["total"])])
            bar_html = bar_chart.to_html(full_html=False)
        else:
            bar_html = "<p>No hay columna 'fecha'.</p>"

        return render_template("dashboard.html", pie_html=pie_html, bar_html=bar_html)

    except Exception as e:
        return f"Error al generar dashboard: {str(e)}"


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    app.run(debug=True)
