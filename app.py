import io, os, base64, tempfile, cv2, numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from inference_sdk import InferenceHTTPClient
from supabase import create_client, Client
from datetime import datetime


app = Flask(__name__)
CORS(app)

# Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ.get("ROBOFLOW_API_KEY")
)

FRUIT_MODELS = {
    "mango": "mango-bzoww/1",
    "strawberry": "strawberry-p7nq9/2",
    "pineapple": "pineapple-xooc7/1",
    "banana": "banana-gh2yn/1",
    "orange": "orange-jsuej/1",
    "watermelon": "watermelon-xztju/5"
}

MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "pineapple-detector/1")

GENERAL_MODEL = "fruits-and-vegetables-yz9mm/1"  # 👈 modelo general de frutas

# Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_API_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route("/")
def index():
    try:
        response = supabase.table("predictions").select("*").execute()
        data = response.data or []

        from collections import defaultdict

        conteo = defaultdict(int)
        for fila in data:
            for pred in fila.get("predicciones", []):
                clase = pred.get("class")
                if clase:
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
def process_auto():
    try:
        image_file = request.files.get("imageFile")
        if not image_file:
            return jsonify({"error": "No se recibió la imagen"}), 400

        image_bytes = image_file.read()

        # === 1️⃣ PRIMERA ETAPA: detectar tipo de fruta ===
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            tmp.write(image_bytes)
            tmp.flush()
            fruit_result = CLIENT.infer(tmp.name, model_id=GENERAL_MODEL)

        fruit_class = fruit_result["predictions"][0]["class"].lower()
        if fruit_class in ["orange"]:
            fruit_class = "naranja"
        elif fruit_class in ["pineapple"]:
            fruit_class = "pina"
        elif fruit_class in ["strawberry"]:
            fruit_class = "fresa"
        elif fruit_class in ["watermelon"]:
            fruit_class = "sandia"
        elif fruit_class in ["mango"]:
            fruit_class = "mango"
            
        model_id = FRUIT_MODELS.get(fruit_class, "pineapple-detector/1")

        print(f"➡️ Detectado: {fruit_class} | Usando modelo: {model_id}")

        # === 2️⃣ SEGUNDA ETAPA: análisis de calidad con el modelo correcto ===
        with tempfile.NamedTemporaryFile(delete=True) as tmp2:
            tmp2.write(image_bytes)
            tmp2.flush()
            quality_result = CLIENT.infer(tmp2.name, model_id=model_id)

        # Guardar en Supabase
        # === Normalizar predicciones antes de guardar ===
        preds = quality_result.get("predictions", [])
        normalized_preds = []
        
        for p in preds:
            clase = p.get("class", "").lower().strip()
        
            # Unificar nombres variantes
            if clase in ["unripen", "unripe", "green", "immature"]:
                clase = "unripe"
            elif clase in ["ripe", "ripe-orange", "ripen"]:
                clase = "ripe"
            elif clase in ["overripe", "too-ripe", "rotten"]:
                clase = "overripe"
        
            p["class"] = clase
            normalized_preds.append(p)
        
        # Guardar en Supabase ya normalizado
        supabase.table("predictions").insert({
            "fecha": datetime.now().isoformat(),
            "imagen": image_file.filename,
            "fruta": fruit_class,
            "modelo": model_id,
            "predicciones": normalized_preds
        }).execute()

        # Devolver imagen procesada y resultado
        img = Image.open(io.BytesIO(image_bytes))
        img = np.array(img)
        _, img_encoded = cv2.imencode(".png", img)
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")

        return jsonify({
            "image": img_base64,
            "fruta_detectada": fruit_class,
            "modelo_usado": model_id,
            "json": quality_result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/graficos")
def graficos():
    try:
        response = supabase.table("predictions").select("*").execute()
        data = response.data or []

        from collections import defaultdict
        conteo = defaultdict(int)

        for fila in data:
            for pred in fila.get("predicciones", []):
                clase = pred.get("class", "").lower().strip()
                if clase in ["unripen", "unripened"]:
                    clase = "unripe"
                elif clase in ["ripen", "ripe"]:
                    clase = "ripe"
                elif clase in ["overripen", "overripe"]:
                    clase = "overripe"
                conteo[clase] += 1

        labels = list(conteo.keys())
        valores = list(conteo.values())

        return jsonify({
            "labels": labels,
            "values": valores
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/process/<fruta>", methods=["POST"])
def process_fruit(fruta):
    try:
        model_id = FRUIT_MODELS.get(fruta, MODEL_ID)
        image_file = request.files.get("imageFile")
        if not image_file:
            return jsonify({"error": "No se recibió la imagen"}), 400

        image_bytes = image_file.read()
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            tmp.write(image_bytes)
            tmp.flush()
            result = CLIENT.infer(tmp.name, model_id=model_id)

        supabase.table("predictions").insert({
            "fecha": datetime.now().isoformat(),
            "imagen": image_file.filename,
            "fruta": fruta,
            "modelo": model_id,
            "predicciones": result.get("predictions", [])
        }).execute()

        _, img_encoded = cv2.imencode(".png", cv2.cvtColor(np.array(Image.open(io.BytesIO(image_bytes))), cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")
        return jsonify({"image": img_base64, "json": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/datos")
def datos():
    try:
        response = supabase.table("predictions").select("predicciones").execute()
        registros = response.data
        return jsonify(registros)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
