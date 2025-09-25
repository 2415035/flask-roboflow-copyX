import io, os, base64, tempfile, cv2, numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from inference_sdk import InferenceHTTPClient
from supabase import create_client, Client
from datetime import datetime

app = Flask(__name__)

# Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ.get("ROBOFLOW_API_KEY")
)
MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "pineapple-detector/1")

# Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_API_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route("/")
def index():
    try:
        response = supabase.table("predictions").select("*").execute()
        data = response.data or []

        conteo = {"ripen": 0, "unripe": 0}
        for fila in data:
            for pred in fila.get("predicciones", []):
                clase = pred.get("class")
                if clase in conteo:
                    conteo[clase] += 1

        import plotly.express as px
        import pandas as pd

        df = pd.DataFrame(list(conteo.items()), columns=["clase", "cantidad"])

        fig_bar = px.bar(df, x="clase", y="cantidad", title="Conteo de predicciones")
        fig_pie = px.pie(df, names="clase", values="cantidad", title="Distribución")

        return render_template("index.html",
                               graph_bar=fig_bar.to_html(full_html=False),
                               graph_pie=fig_pie.to_html(full_html=False))
    except Exception as e:
        return f"<h2>Error cargando gráficos: {str(e)}</h2>"

@app.route("/process", methods=["POST"])
def process():
    try:
        image_file = request.files.get("imageFile")
        if not image_file:
            return jsonify({"error": "No se recibió la imagen"}), 400

        image_bytes = image_file.read()
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            tmp.write(image_bytes)
            tmp.flush()
            result = CLIENT.infer(tmp.name, model_id=MODEL_ID)

        # Convertir a numpy para mostrar
        img = Image.open(io.BytesIO(image_bytes))
        img = np.array(img)

        for pred in result.get("predictions", []):
            x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
            label = pred["class"]
            conf = pred["confidence"]
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Guardar en Supabase
        supabase.table("predictions").insert({
            "fecha": datetime.now().isoformat(),
            "imagen": image_file.filename,
            "predicciones": result.get("predictions", [])
        }).execute()

        _, img_encoded = cv2.imencode(".png", img)
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")

        return jsonify({ "image": img_base64, "json": result })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
