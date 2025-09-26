import logging
import time
from flask import request
from flask_restx import Namespace, Resource, fields
from src.core.objective_parser import ObjectiveParser

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
# Create Objective Parser Object
# ======================================================
objective_parser = ObjectiveParser(logger=logger)

@api.route("/extract/")
class Extract(Resource):
    @api.expect(objective_extractor_model)  # This tells Swagger to expect JSON input
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

            input_text = data["text"].strip()
            
            logger.info(f"Input text: {input_text}")

            if not input_text:
                return {"error": "Invalid input. Please provide non-empty text."}, 400

            # Extract objectives from text
            extracted_objectives = objective_parser.parse(input_text)

            end_time = time.time() - start_time
            response = {
                "responseHeader": {"status": 200, "time": end_time},
                "response": extracted_objectives,
            }
            logger.info("Objectives extracted successfully")

            return response, 200

        except Exception as e:
            logger.error(f"Error extracting objectives: {str(e)}")
            return {"error": "Failed to process the text", "details": str(e)}, 500


"""
@api.route("/extract/")
class extract(Resource):
    @api.doc(
        parser=text_parser,
        responses={
            200: "Success: Objectives extracted successfully",
            400: "Bad Request: Invalid input",
            500: "Server Error: Failed to process the text",
        },
    )
    def post(self):
        start_time = time.time()
        args = text_parser.parse_args()
        input_text = args["text"]

        if not input_text.strip():
            return {"error": "Invalid input. Please provide non-empty text."}, 400

        try:
            # Extract objectives from text
            extracted_objectives = objective_parser.parse(input_text)

            end_time = time.time() - start_time
            response = {
                "responseHeader": {"status": 200, "time": end_time},
                "response": extracted_objectives,
            }
            logger.info("Objectives extracted successfully")
            
            return response, 200

        except Exception as e:
            logger.error(f"Error extracting objectives: {str(e)}")
            return {"error": "Failed to process the text", "details": str(e)}, 500
"""