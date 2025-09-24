from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
import plotly.graph_objs as go
import base64
from datetime import datetime

app = Flask(__name__)

# === SUPABASE ===
SUPABASE_URL = "https://ipgfsxnaohvsnnoxhdbv.supabase.co"
SUPABASE_TABLE = "predictions"

# ⚠️ Usa tu API KEY real aquí (la anon) o mejor por variables de entorno en Render
SUPABASE_API_KEY = "TU_API_KEY_AQUI"

HEADERS = {
    "apikey": SUPABASE_API_KEY,
    "Authorization": f"Bearer {SUPABASE_API_KEY}",
}

# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------
def supabase_select_all():
    """Devuelve lista de dicts. Tolera 1 fila (dict), errores y respuestas vacías."""
    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}?select=*"
    r = requests.get(url, headers=HEADERS)
    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"Respuesta no JSON de Supabase: status={r.status_code} body={r.text[:300]}")

    # Si llega un dict (una fila) o un error
    if isinstance(data, dict):
        # error típico de PostgREST
        if any(k in data for k in ("message", "hint", "details", "code", "error")):
            raise RuntimeError(f"Error Supabase: {data}")
        # una única fila → lo volvemos lista
        data = [data]

    if not isinstance(data, list):
        raise RuntimeError(f"Formato inesperado de Supabase (no lista): {type(data)} => {data}")

    return data


def to_dataframe(rows):
    """Crea un DataFrame estable a partir de lista de dicts."""
    if not rows:
        return pd.DataFrame()
    # json_normalize maneja campos anidados como predicciones
    df = pd.json_normalize(rows)
    return df


# ------------------------------------------------------------------------------
# rutas
# ------------------------------------------------------------------------------
@app.route("/")
def home():
    # En tu ZIP existe templates/index.html (no upload.html)
    return render_template("index.html", title="Subir imagen")

@app.route("/process", methods=["POST"])
def process():
    """
    Simulación de procesamiento para que el flujo funcione.
    Tu modelo real debería generar 'predictions' y (opcional) la imagen procesada.
    """
    file = request.files.get("imageFile")
    if not file:
        return jsonify({"error": "No se envió archivo"}), 400

    img_b64 = base64.b64encode(file.read()).decode("utf-8")

    result_json = {
        "predictions": [
            {"class": "ripen", "confidence": 0.92},
            {"class": "unripen", "confidence": 0.44},
        ]
    }

    return jsonify({"image": img_b64, "json": result_json})

@app.route("/save_prediction", methods=["POST"])
def save_prediction():
    try:
        payload = request.json or {}

        # Si no viene fecha desde el frontend, poner ahora (ISO)
        payload.setdefault("fecha", datetime.utcnow().isoformat())

        url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
        r = requests.post(
            url,
            headers={**HEADERS, "Content-Type": "application/json"},
            json=payload,
        )

        # 201 o 200 según PostgREST
        if r.status_code in (200, 201):
            return jsonify({"status": "ok", "data": r.json()})
        else:
            # Pasar error textual para depurar en pantalla
            return jsonify({"status": "error", "msg": r.text}), 400
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route("/dashboard")
def dashboard():
    try:
        rows = supabase_select_all()         # <- aquí normalizamos la respuesta
        if not rows:
            return "No hay datos disponibles desde Supabase."

        df = to_dataframe(rows)
        if df.empty:
            return "No hay datos para mostrar."

        # ---------- columnas mínimas ----------
        # fecha
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
        else:
            # si no existe, asumir hoy (para no romper gráficos)
            df["fecha"] = pd.to_datetime(datetime.utcnow(), utc=True).date()

        # claseValidada
        if "claseValidada" not in df.columns:
            # intenta derivar de 'predicciones' (si existe) la clase top
            if "predicciones" in df.columns:
                # cuando Supabase devuelve JSONB → suele ser dict/list; ya está como objeto Python en df
                def top_cls(p):
                    try:
                        if isinstance(p, list) and p:
                            # devolver la clase con mayor confianza
                            best = max(p, key=lambda x: x.get("confidence", x.get("score", 0)))
                            return best.get("class") or best.get("label") or best.get("name")
                    except Exception:
                        return None
                    return None
                df["claseValidada"] = df["predicciones"].apply(top_cls)
            else:
                df["claseValidada"] = "sin_clase"

        df["claseValidada"] = df["claseValidada"].fillna("sin_clase")

        # ---------- Gráfico de torta (clases) ----------
        pie_counts = df["claseValidada"].value_counts(dropna=False).reset_index()
        pie_counts.columns = ["label", "value"]
        pie_chart = go.Figure(data=[go.Pie(labels=pie_counts["label"], values=pie_counts["value"])])
        pie_chart.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        pie_html = pie_chart.to_html(full_html=False)

        # ---------- Gráfico de barras (por fecha) ----------
        bar_data = (
            df.groupby("fecha", dropna=False).size().reset_index(name="total").sort_values("fecha")
        )
        bar_chart = go.Figure(data=[go.Bar(x=bar_data["fecha"], y=bar_data["total"])])
        bar_chart.update_layout(xaxis_title="Fecha", yaxis_title="Total", margin=dict(t=20, b=20, l=20, r=20))
        bar_html = bar_chart.to_html(full_html=False)

        return render_template("dashboard.html", pie_html=pie_html, bar_html=bar_html)

    except Exception as e:
        return f"Error al generar dashboard: {str(e)}"

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
