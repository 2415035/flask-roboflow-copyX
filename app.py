from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
import plotly.graph_objs as go

app = Flask(__name__)

# === Configuración de InstantDB ===
INSTANTDB_API_URL = "https://instantdb.io/api/project/proyecto-productivo/predictions"
HEADERS = {
    "X-INSTANT-API-KEY": "646b5cf0-25ff-4084-9ed9-505666a1bb1a"
}

@app.route("/")
def home():
    return render_template('upload.html', title='Subir imagen')

@app.route("/dashboard")
def dashboard():
    try:
        response = requests.get(INSTANTDB_API_URL, headers=HEADERS)
        data = response.json()

        if "data" not in data:
            return "Error: No se encontraron datos en la respuesta de InstantDB."

        df = pd.DataFrame(data["data"])

        if df.empty:
            return "No hay datos para mostrar."

        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
        else:
            return "Error: No se encontró la columna 'fecha'."

        if "claseValidada" not in df.columns:
            return "Error: No se encontró la columna 'claseValidada'."
        
        pie_data = df["claseValidada"].value_counts().reset_index()
        pie_chart = go.Figure(data=[go.Pie(labels=pie_data['index'], values=pie_data['claseValidada'])])
        pie_html = pie_chart.to_html(full_html=False)

        bar_data = df.groupby("fecha").size().reset_index(name="total")
        bar_chart = go.Figure(data=[go.Bar(x=bar_data["fecha"], y=bar_data["total"])])
        bar_html = bar_chart.to_html(full_html=False)

        return render_template("dashboard.html", pie_html=pie_html, bar_html=bar_html)

    except Exception as e:
        return f"Error al generar dashboard: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
