from flask import Flask, render_template, request
import pandas as pd
import pickle
from pathlib import Path

app = Flask(__name__)

CONVERSION = {
    "Nunca": 0,
    "Casi nunca": 1,
    "A veces": 2,
    "Casi siempre": 3,
    "Siempre": 4,
}

DEFAULT_QUESTIONS = [
    "Mi pareja me pide que le envíe mi ubicación constantemente",
    "Me revisa el celular o redes sociales",
    "Me dice con quién puedo o no puedo hablar",
    "Se molesta si no respondo mensajes inmediatamente",
    "Me ha pedido que deje de hablar con ciertas amistades",
    "Mi pareja se enoja cuando convivo con otras personas",
    "Me cuestiona constantemente sobre dónde estoy",
    "Interpreta cosas neutras como infidelidad",
    "Me hace sentir culpable por salir sin ella/él",
    "Siento que no puedo terminar la relación aunque no esté bien",
    "Tengo miedo de estar sola/o",
    "Prefiero evitar conflictos aunque me incomode",
    "Cambié cosas importantes de mí por la relación",
    "Es normal que las parejas se revisen el celular",
    "Los celos son prueba de amor",
    "Es mejor ceder para evitar problemas",
    "A veces el control es por cuidado",
    "Me siento libre de tomar decisiones",
    "Mantengo mis amistades",
    "Tengo actividades propias fuera de la relación",
]

def load_bundle():
    with open("modelo.pkl", "rb") as f:
        bundle = pickle.load(f)

    if isinstance(bundle, dict) and "model" in bundle:
        return bundle

    # Compatibilidad por si existe un modelo viejo guardado solo como objeto
    return {
        "model": bundle,
        "feature_columns": DEFAULT_QUESTIONS.copy(),
        "threshold": 0,
        "reverse_indexes": [17, 18, 19],
    }

BUNDLE = load_bundle()
MODEL = BUNDLE["model"]
FEATURE_COLUMNS = BUNDLE["feature_columns"]
THRESHOLD = BUNDLE.get("threshold", 0)
REVERSE_INDEXES = BUNDLE.get("reverse_indexes", [17, 18, 19])

def prepare_input(raw_values):
    values = list(raw_values)

    # Ajuste de preguntas con sentido inverso
    for idx in REVERSE_INDEXES:
        if idx < len(values):
            values[idx] = 4 - values[idx]

    return pd.DataFrame([values], columns=FEATURE_COLUMNS)

def load_study_stats():
    excel_path = Path("encuesta.xlsx")
    if not excel_path.exists():
        return {
            "total": 0,
            "threshold": 0,
            "distribution": [
                {"label": "Bajo", "value": 0},
                {"label": "Medio", "value": 0},
                {"label": "Alto", "value": 0},
            ],
            "top_factors": [],
        }

    raw = pd.read_excel(excel_path)
    raw.columns = [str(c).strip() for c in raw.columns]

    drop_cols = ["Marca temporal", "Nombre:"]
    for col in drop_cols:
        if col in raw.columns:
            raw = raw.drop(columns=[col])

    if all(q in raw.columns for q in FEATURE_COLUMNS):
        data = raw[FEATURE_COLUMNS].copy()
    else:
        data = raw.iloc[:, :len(FEATURE_COLUMNS)].copy()
        data.columns = FEATURE_COLUMNS

    data = data.replace(CONVERSION)
    data = data.apply(pd.to_numeric, errors="coerce").fillna(0)

    for idx in REVERSE_INDEXES:
        if idx < len(FEATURE_COLUMNS):
            data.iloc[:, idx] = 4 - data.iloc[:, idx]

    total = len(data)
    scores = data.sum(axis=1)

    q1 = scores.quantile(0.33)
    q2 = scores.quantile(0.66)

    low = round((scores <= q1).mean() * 100)
    medium = round(((scores > q1) & (scores <= q2)).mean() * 100)
    high = round((scores > q2).mean() * 100)

    factor_scores = (data.mean(axis=0) / 4.0 * 100).round(0)
    top_factors = (
        factor_scores.sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    top_factors.columns = ["label", "value"]

    return {
        "total": total,
        "threshold": round(float(THRESHOLD), 2) if THRESHOLD else 0,
        "distribution": [
            {"label": "Bajo", "value": low},
            {"label": "Medio", "value": medium},
            {"label": "Alto", "value": high},
        ],
        "top_factors": top_factors.to_dict(orient="records"),
    }

STATS = load_study_stats()

@app.route("/")
def inicio():
    return render_template(
        "index.html",
        questions=FEATURE_COLUMNS,
        total_responses=STATS["total"],
        threshold=STATS["threshold"],
        top_factors=STATS["top_factors"],
        distribution=STATS["distribution"],
        nivel=None,
        porcentaje=None,
        angle=None,
        color="#4f46e5",
        mensaje="",
        factors=[],
        decision=None,
    )

@app.route("/predecir", methods=["POST"])
def predecir():
    values = []
    raw_values = []

    for i in range(len(FEATURE_COLUMNS)):
        val = int(request.form.get(f"p{i}", 0))
        raw_values.append(val)
        values.append(val)

    datos = prepare_input(values)

    prob = float(MODEL.predict_proba(datos)[0][1])
    porcentaje = round(prob * 100)
    angle = round(porcentaje * 3.6, 2)

    if porcentaje >= 70:
        nivel = "RIESGO ALTO"
        color = "#ef4444"
        decision = "SÍ presenta indicadores de violencia"
        mensaje = (
            "Las respuestas muestran indicadores importantes de control, "
            "dependencia emocional y conductas asociadas a violencia en el noviazgo."
        )
    elif porcentaje >= 40:
        nivel = "RIESGO MEDIO"
        color = "#f59e0b"
        decision = "Posible presencia de riesgo"
        mensaje = (
            "Existen algunas conductas de alerta. Conviene observar la relación "
            "y fortalecer la comunicación y los límites personales."
        )
    else:
        nivel = "RIESGO BAJO"
        color = "#22c55e"
        decision = "NO presenta indicadores importantes"
        mensaje = (
            "Las respuestas muestran pocos indicadores relacionados con violencia en el noviazgo. "
            "Aun así, es importante mantener respeto, confianza y libertad."
        )

    detected = []
    for idx, (question, raw) in enumerate(zip(FEATURE_COLUMNS, raw_values)):
        if idx in REVERSE_INDEXES:
            if raw <= 1:
                detected.append(question)
        else:
            if raw >= 3:
                detected.append(question)

    detected = detected[:5]

    quote = "Una relación sana se construye con respeto, confianza y libertad personal."

    return render_template(
        "index.html",
        questions=FEATURE_COLUMNS,
        total_responses=STATS["total"],
        threshold=STATS["threshold"],
        top_factors=STATS["top_factors"],
        distribution=STATS["distribution"],
        nivel=nivel,
        porcentaje=porcentaje,
        angle=angle,
        color=color,
        mensaje=mensaje,
        factors=detected,
        decision=decision,
        quote=quote,
    )

if __name__ == "__main__":
    app.run(debug=True)