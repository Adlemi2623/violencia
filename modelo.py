import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from pathlib import Path

EXCEL_FILE = "encuesta.xlsx"

# Preguntas base del formulario
QUESTIONS = [
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

CONVERSION = {
    "Nunca": 0,
    "Casi nunca": 1,
    "A veces": 2,
    "Casi siempre": 3,
    "Siempre": 4,
}

# Preguntas con sentido inverso: respuestas altas aquí indican menos riesgo
REVERSE_INDEXES = [17, 18, 19]  # 18, 19 y 20 en lenguaje humano

DROP_COLS = ["Marca temporal", "Nombre:"]

def normalize_text(s):
    return str(s).strip()

def prepare_features(df, feature_columns):
    data = df.copy()
    data.columns = [normalize_text(c) for c in data.columns]

    for col in DROP_COLS:
        if col in data.columns:
            data = data.drop(columns=[col])

    if all(q in data.columns for q in feature_columns):
        data = data[feature_columns].copy()
    else:
        # Si los nombres no coinciden exactamente, tomar las primeras 20 columnas útiles
        data = data.iloc[:, :len(feature_columns)].copy()
        data.columns = feature_columns

    data = data.replace(CONVERSION)
    data = data.apply(pd.to_numeric, errors="coerce").fillna(0)

    for idx in REVERSE_INDEXES:
        if idx < len(feature_columns):
            data.iloc[:, idx] = 4 - data.iloc[:, idx]

    return data

def main():
    excel_path = Path(EXCEL_FILE)
    if not excel_path.exists():
        raise FileNotFoundError(
            f"No se encontró '{EXCEL_FILE}' en la carpeta del proyecto."
        )

    df = pd.read_excel(excel_path)
    df.columns = [normalize_text(c) for c in df.columns]

    # Usar las preguntas reales si están; si no, tomar las primeras 20 columnas útiles
    if all(q in df.columns for q in QUESTIONS):
        feature_columns = QUESTIONS.copy()
    else:
        feature_columns = [c for c in df.columns if c not in DROP_COLS][:20]

    if len(feature_columns) < 20:
        raise ValueError("No se detectaron suficientes columnas de preguntas en el Excel.")

    X = prepare_features(df, feature_columns)

    # Crear una etiqueta binaria de riesgo (0 = No, 1 = Sí) a partir del puntaje total
    total_scores = X.sum(axis=1)
    threshold = float(total_scores.median())
    y = (total_scores >= threshold).astype(int)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X, y)

    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "threshold": threshold,
        "reverse_indexes": REVERSE_INDEXES,
    }

    with open("modelo.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print("Modelo entrenado correctamente")
    print(f"Columnas usadas: {len(feature_columns)}")
    print(f"Umbral de riesgo: {threshold:.2f}")

if __name__ == "__main__":
    main()