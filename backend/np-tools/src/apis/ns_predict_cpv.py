# src/api/ns_predict_cpv8.py

import time
import logging
from typing import List
from flask import request
from flask_restx import Namespace, Resource, reqparse
from src.core.cpv_classifier.cpv_classifier_8 import CPV8ClassifierHF

HF_REPO_ID = "erick4556/cpv8-bert-spanish-ft"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CPV8Predict")
logger.setLevel(logging.INFO)

api = Namespace(name="Cpv Prediction - 8 digits")

cpv_parser = reqparse.RequestParser()
cpv_parser.add_argument(
    "texts", help="Text(s) to predict CPV-8 codes", action="split", required=False)

# Inicializa el modelo una sola vez
try:
    classifier = CPV8ClassifierHF(
        repo_id=HF_REPO_ID,
        device=None,          
        default_threshold=0.80,
        hf_local_dir="/models/cpv8"
    )
    logger.info("CPV8ClassifierHF inicializado correctamente.")
except Exception as e:
    logger.exception(f"Error inicializando CPV8ClassifierHF: {e}")
    classifier = None


def _extract_texts_from_request() -> List[str]:
    # JSON: {"texts": [...]}
    if request.is_json:
        data = request.get_json(silent=True) or {}
        texts = data.get("texts")
        if isinstance(texts, list):
            return [str(t) for t in texts if str(t).strip()]
        elif isinstance(texts, str) and texts.strip():
            return [texts.strip()]
    # form-data/query
    args = cpv_parser.parse_args()
    texts = args.get("texts")
    if isinstance(texts, list) and len(texts) > 0:
        return [str(t).strip() for t in texts if str(t).strip()]
    return []


@api.route("/predict")
class Predict(Resource):
    @api.doc(
        parser=cpv_parser,
        responses={
            200: "Success",
            400: "Bad Request",
            502: "Model not available or prediction error",
        },
    )
    def post(self):
        start_time = time.time()

        if classifier is None:
            return {"status": 502, "error": "Modelo no inicializado."}, 502

        texts = _extract_texts_from_request()
        if not texts:
            return {"status": 400, "error": "Debes enviar 'texts' como lista en JSON o form-data."}, 400

        try:
            results = classifier.predict_batch(texts)
            payload = [
                {
                    "text": r.get("original_text", ""),
                    "cpv_code": r.get("cpv_predicted", None)
                }
                for r in results
            ]
            return {
                "status": 200,
                "count": len(payload),
                "execution_time_seconds": round(time.time() - start_time, 3),
                "predictions": payload
            }, 200
        except Exception as e:
            logger.exception(f"Error en predicción CPV-8: {e}")
            return {
                "status": 502,
                "error": str(e),
                "execution_time_seconds": round(time.time() - start_time, 3),
            }, 502