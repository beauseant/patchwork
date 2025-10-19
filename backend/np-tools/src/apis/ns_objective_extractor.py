import logging
import time
from flask import request # type: ignore
from flask_restx import Namespace, Resource, fields # type: ignore
from src.core.objective_extractor.extract import ObjectiveExtractor 

# ======================================================
# Logging Configuration
# ======================================================
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ObjectiveExtractor")

# ======================================================
# Define Namespace for Objective Extraction
# ======================================================
api = Namespace("Objective Extraction")

# ======================================================
# Define API Model for Input (Swagger Documentation)
# ======================================================
objective_extractor_model = api.model("ObjectiveExtractor", {
    "text": fields.String(required=True, description="Text from which to extract objectives")
})

# ======================================================
# Create Objective Extractor Object (heavy init once)
# ======================================================
extractor = ObjectiveExtractor(
    logger=logger,
    config_path="/np-tools/src/core/objective_extractor/config/config.yaml"
)

@api.route("/extract/")
class Extract(Resource):
    @api.expect(objective_extractor_model)
    @api.doc(
        responses={
            200: "Success: Objectives extracted successfully",
            400: "Bad Request: Invalid input",
            500: "Server Error: Failed to process the text",
        },
    )
    def post(self):
        start_time = time.time()

        try:
            # Read JSON payload
            data = request.get_json()

            if not data or "text" not in data:
                return {"error": "Invalid input. Please provide a 'text' field in JSON."}, 400

            logger.info("Received input for objective extraction")

            input_text = (data["text"] or "").strip()

            # avoid logging entire long payloads
            logger.debug(f"Input text (preview): {input_text[:500]}{'...' if len(input_text) > 500 else ''}")

            if not input_text:
                return {"error": "Invalid input. Please provide non-empty text."}, 400

            extracted_objectives = {"generated_objective": extractor.extract_generative(input_text)}

            end_time = time.time() - start_time
            response = {
                "responseHeader": {"status": 200, "time": end_time},
                "response": extracted_objectives,
            }
            logger.info("Objectives extracted successfully")

            return response, 200

        except Exception as e:
            logger.error(f"Error extracting objectives: {str(e)}", exc_info=True)
            return {"error": "Failed to process the text", "details": str(e)}, 500