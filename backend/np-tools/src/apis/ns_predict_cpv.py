# src/api/ns_predict_cpv8.py

import time
import logging
from typing import List
from flask import request
from flask_restx import Namespace, Resource, fields  # type: ignore
from src.core.cpv_classifier.cpv_classifier_8 import CPV8ClassifierHF

HF_REPO_ID = "erick4556/cpv8-bert-spanish-ft"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CPV8Predict")
logger.setLevel(logging.INFO)

# ======================================================
# Define Namespace for CPV prediction
# ======================================================
api = Namespace(name="Cpv Prediction - 8 digits")


# ======================================================
# Define API Model for Input (Swagger Documentation)
# ======================================================
cpv_predictor_model = api.model("CPVPredictor", {
    "text": fields.String(required=True, description="Text from which to extract CPV-8 codes")
})

# ======================================================
# Create CPV Predictor Object (heavy init once)
# ======================================================
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


@api.route("/predict")
class Predict(Resource):
    @api.expect(cpv_predictor_model)
    @api.doc(
        responses={
            200: "Success: CPV codes predicted successfully",
            400: "Bad Request",
            502: "Model not available or prediction error",
        },
    )
    def post(self):
        start_time = time.time()

        if classifier is None:
            return {"error": "Modelo no inicializado."}, 502

        try:
            # Read JSON payload
            data = request.get_json()

            if not data or "text" not in data:
                return {"error": "Invalid input. Please provide a 'text' field in JSON."}, 400

            logger.info("Received input for CPV prediction")

            input_text = (data["text"] or "").strip()

            # avoid logging entire long payloads
            logger.debug(f"Input text (preview): {input_text[:500]}{'...' if len(input_text) > 500 else ''}")

            if not input_text:
                return {"error": "Invalid input. Please provide non-empty text."}, 400

            results = classifier.predict_one(input_text)
            prediction_results = {
                "text": results.get("original_text", ""),
                "cpv_code": results.get("cpv_predicted", None),
                "probability": results.get("prob", None)
            }

            end_time = time.time() - start_time
            response = {
                "responseHeader": {"status": 200, "time": end_time},
                "response": prediction_results,
            }
            logger.info("CPV prediction completed successfully")

            return response, 200

        except Exception as e:
            logger.error(f"Error during CPV prediction: {str(e)}", exc_info=True)
            return {"error": "Failed to process the text", "details": str(e)}, 500
