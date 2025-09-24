from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
import plotly.graph_objs as go
import base64
import io

app = Flask(__name__)

# === InstantDB credentials ===
INSTANTDB_API_URL = "https://app.instantdb.io/api/project/proyecto-productivo/predictions"
HEADERS = {
    "X-INSTANT-API-KEY": "646b5cf0-25ff-4084-9ed9-505666a1bb1a"
}

@app.route("/")
def home():
    return render_template('upload.html', title='Subir imagen')

@app.route("/dashboard")
def dashboard():
    # 🔹 Obtener datos de InstantDB
    response = requests.get(INSTANTDB_API_URL, headers=HEADERS)
    data = response.json()

    # 🔹 Convertir a DataFrame
    df = pd.DataFrame(data['data'])

    # 🔹 Gráfico de torta (por claseValidada)
    pie_data = df['claseValidada'].value_counts().reset_index()
    pie_chart = go.Figure(data=[go.Pie(labels=pie_data['index'], values=pie_data['claseValidada'])])
    pie_html = pie_chart.to_html(full_html=False)

    # 🔹 Gráfico de barras (por fecha)
    bar_data = df.groupby('fecha').size().reset_index(name='total')
    bar_chart = go.Figure(data=[go.Bar(x=bar_data['fecha'], y=bar_data['total'])])
    bar_html = bar_chart.to_html(full_html=False)

    return render_template("dashboard.html", pie_html=pie_html, bar_html=bar_html)

if __name__ == "__main__":
    app.run(debug=True)
