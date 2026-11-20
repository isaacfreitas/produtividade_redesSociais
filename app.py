"""Aplicação web Flask para inferência com o modelo Naive Bayes treinado."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "modelo_naive_bayes.pkl"

NUMERIC_FEATURES = [
    "age",
    "daily_social_media_time",
    "number_of_notifications",
    "work_hours_per_day",
    "stress_level",
    "sleep_hours",
    "screen_time_before_sleep",
    "breaks_during_work",
    "days_feeling_burnout_per_month",
    "weekly_offline_hours",
    "job_satisfaction_score",
]
CATEGORICAL_FEATURES = ["job_type"]
BOOLEAN_FEATURES = ["uses_focus_apps", "has_digital_wellbeing_enabled"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

BOOLEAN_MAP = {
    "true": True,
    "1": True,
    "yes": True,
    "sim": True,
    "false": False,
    "0": False,
    "no": False,
    "não": False,
    "nao": False,
}

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
logging.basicConfig(level=logging.INFO)


def load_model_artifact() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        logging.warning("Modelo não encontrado em %s. Execute train.py primeiro.", MODEL_PATH)
        return None
    try:
        artifact = joblib.load(MODEL_PATH)
        if "pipeline" not in artifact or "features" not in artifact:
            raise ValueError("O arquivo do modelo não contém os metadados esperados.")
        return artifact
    except Exception:
        logging.exception("Não foi possível carregar o modelo salvo.")
        return None


ARTIFACT = load_model_artifact()


def parse_boolean(value: str, field_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in BOOLEAN_MAP:
        raise ValueError(f"Selecione Sim ou Não para o campo '{field_name}'.")
    return BOOLEAN_MAP[normalized]


def parse_form_data(form: Any) -> pd.DataFrame:
    values: dict[str, Any] = {}

    for feature in NUMERIC_FEATURES:
        raw_value = form.get(feature, "").strip()
        if not raw_value:
            raise ValueError(f"Preencha o campo '{feature.replace('_', ' ')}'.")
        try:
            number = float(raw_value.replace(",", "."))
        except ValueError as exc:
            raise ValueError(f"O campo '{feature.replace('_', ' ')}' deve ser numérico.") from exc
        if not math.isfinite(number):
            raise ValueError(f"O valor informado em '{feature.replace('_', ' ')}' é inválido.")
        values[feature] = number

    job_type = form.get("job_type", "").strip()
    if not job_type:
        raise ValueError("Informe o tipo de ocupação.")
    values["job_type"] = job_type

    for feature in BOOLEAN_FEATURES:
        values[feature] = parse_boolean(form.get(feature, ""), feature)

    return pd.DataFrame([values], columns=ALL_FEATURES)


def probability_payload(row: pd.DataFrame) -> dict[str, Any]:
    assert ARTIFACT is not None
    pipeline = ARTIFACT["pipeline"]
    predicted_class = str(pipeline.predict(row)[0])
    probabilities = pipeline.predict_proba(row)[0]
    model_classes = [str(label) for label in pipeline.classes_]

    probabilities_by_class = {
        label: round(float(probability) * 100, 2)
        for label, probability in zip(model_classes, probabilities)
    }
    predicted_probability = probabilities_by_class.get(predicted_class, 0.0)

    return {
        "prediction": predicted_class,
        "probability": predicted_probability,
        "probabilities": probabilities_by_class,
        "message": "Previsão calculada pelo modelo Naive Bayes.",
    }


@app.get("/")
def home():
    job_types = ARTIFACT.get("job_type_options", []) if ARTIFACT else []
    thresholds = ARTIFACT.get("target_thresholds", {}) if ARTIFACT else {}
    return render_template(
        "index.html",
        model_ready=ARTIFACT is not None,
        job_types=job_types,
        thresholds=thresholds,
    )


@app.post("/predict")
def predict():
    if ARTIFACT is None:
        error = "Modelo ainda não encontrado. Execute 'python train.py' antes de iniciar a aplicação."
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"error": error}), 503
        return render_template(
            "index.html", model_ready=False, job_types=[], error_message=error
        ), 503

    try:
        row = parse_form_data(request.form)
        result = probability_payload(row)
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify(result)

        return render_template(
            "index.html",
            model_ready=True,
            job_types=ARTIFACT.get("job_type_options", []),
            prediction=result["prediction"],
            probability=result["probability"],
            probabilities=result["probabilities"],
        )
    except ValueError as exc:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"error": str(exc)}), 400
        return render_template(
            "index.html",
            model_ready=True,
            job_types=ARTIFACT.get("job_type_options", []),
            error_message=str(exc),
        ), 400
    except Exception:
        logging.exception("Erro inesperado durante a previsão.")
        error = "Ocorreu um erro ao calcular a previsão. Verifique os dados e tente novamente."
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"error": error}), 500
        return render_template(
            "index.html",
            model_ready=True,
            job_types=ARTIFACT.get("job_type_options", []),
            error_message=error,
        ), 500


if __name__ == "__main__":
    # debug=True é útil no desenvolvimento local; desative em produção.
    app.run(host="127.0.0.1", port=5000, debug=True)
