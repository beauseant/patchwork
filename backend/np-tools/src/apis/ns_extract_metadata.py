import logging
import time
from flask import request  # type: ignore
from werkzeug.datastructures import FileStorage
from flask_restx import Namespace, Resource, fields, reqparse  # type: ignore

from src.core.metadata_extractor.metadata_extractor import MetadataExtractor

# -----------------------------
# Logger
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MetadataExtractorAPI")

# -----------------------------
# Namespace Flask-Restx
# -----------------------------
api = Namespace("Metadata Extraction")

# -----------------------------
# Modelo de entrada (solo 'text')
# -----------------------------
metadata_model = api.model("MetadataRequest", {
    "text": fields.String(required=True, description="Texto a analizar")
})

# ======================================================
# Define parser to accept file uploads
# ======================================================
file_upload_parser = reqparse.RequestParser()
file_upload_parser.add_argument(
    "file", type=FileStorage, location="files", required=True, help="PDF file to extract text from."
)

# ======================================================
# Create Objective Extractor Object (heavy init once)
# ======================================================
extractor = MetadataExtractor(
    ollama_llm="llama3.1",
    ollama_embed_model="mxbai-embed-large",
)

# -----------------------------
# Endpoints
# -----------------------------


@api.route("/extract/fromFile/")
class ExtractMetadataFromFile(Resource):
    @api.doc(
        parser=file_upload_parser,
        responses={
            200: "Success: Metadata extracted successfully",
            400: "Bad Request: Invalid input",
            500: "Server Error: Failed to process the file",
        },
    )
    def post(self):
        start_time = time.time()
        args = file_upload_parser.parse_args()
        uploaded_file = args["file"]

        if not uploaded_file:
            return {"error": "Invalid file."}, 400

        logger.info("Received TEXT file for metadata extraction")

        try:
            # read data from file
            input_text = uploaded_file.read().decode("utf-8")

            if not input_text:
                return {"error": "Invalid input text."}, 400

            logger.debug(
                f"Input text (preview): {input_text[:500]}{'...' if len(input_text) > 500 else ''}")

            result = extractor.extract_metadata_from_text(input_text)

            end_time = time.time() - start_time
            response = {
                "responseHeader": {"status": 200, "time": end_time},
                "response": result,
            }
            logger.info("Metadata extracted successfully")

            return response, 200

        except Exception as e:
            logger.error(
                f"Error extracting metadata: {str(e)}", exc_info=True)
            return {"error": "Failed to process the text", "details": str(e)}, 500


@api.route("/extract/fromText/")
class ExtractMetadataFromText(Resource):
    @api.expect(metadata_model)
    @api.doc(
        responses={
            200: "Success: Metadata extracted successfully",
            400: "Bad Request: Missing or invalid 'text'",
            502: "Metadata extraction error",
        }
    )
    def post(self):
        start_time = time()
        try:
            data = request.get_json(silent=True) or {}
            text = data.get("text", "")

            if not isinstance(text, str) or not text.strip():
                return {"error": "'text' debe ser un string no vacío"}, 400

            result = extractor.extract_metadata_from_text(text)

            end_time = time() - start_time

            # Si hubo error, devolver 502; si no, 200 con las tres claves
            if "error" in result:
                logger.error(f"Metadata extraction error: {result['error']}")
                return {
                    "responseHeader": {"status": 502, "time": end_time},
                    "response": result
                }, 502

            response = {
                "responseHeader": {"status": 200, "time": end_time},
                "response": result
            }
            logger.info("Metadata extracted successfully from text")
            return response, 200

        except Exception as e:
            logger.exception(f"Metadata extraction exception: {e}")
            return {
                "responseHeader": {"status": 502, "time": time() - start_time},
                "response": {"error": str(e)}
            }, 502
