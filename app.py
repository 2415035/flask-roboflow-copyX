from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
import plotly.graph_objs as go

app = Flask(__name__)

# === Configuración de SUPABASE ===
SUPABASE_URL = "https://ipgfsxnaohvsnnoxhdbv.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlwZ2ZzeG5hb2h2c25ub3hoZGJ2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg3Mzc4OTQsImV4cCI6MjA3NDMxMzg5NH0.Ia_CdNAiu5HKPsjc_J5e5Xu1Zoh-bwrBtnpvHP9-D_w"
SUPABASE_TABLE = "predictions"

HEADERS = {
    "apikey": SUPABASE_API_KEY,
    "Authorization": f"Bearer {SUPABASE_API_KEY}"
}

@app.route("/")
def home():
    return render_template('upload.html', title='Subir imagen')

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

        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
        else:
            return "Error: No se encontró la columna 'fecha'."

        if "claseValidada" not in df.columns:
            return "Error: No se encontró la columna 'claseValidada'."
        
        # Gráfico de torta
        pie_data = df["claseValidada"].value_counts().reset_index()
        pie_chart = go.Figure(data=[go.Pie(labels=pie_data['index'], values=pie_data['claseValidada'])])
        pie_html = pie_chart.to_html(full_html=False)

        # Gráfico de barras
        bar_data = df.groupby("fecha").size().reset_index(name="total")
        bar_chart = go.Figure(data=[go.Bar(x=bar_data["fecha"], y=bar_data["total"])])
        bar_html = bar_chart.to_html(full_html=False)

        return render_template("dashboard.html", pie_html=pie_html, bar_html=bar_html)

    except Exception as e:
        return f"Error al generar dashboard: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
