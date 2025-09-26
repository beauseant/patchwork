import logging
from flask import request # type: ignore
from flask_restx import Namespace, Resource, fields # type: ignore

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

# -----------------------------
# Endpoint
# -----------------------------
@api.route("/extract/")
class ExtractMetadata(Resource):
    @api.expect(metadata_model)
    @api.doc(
        responses={
            200: "Success: Metadata extracted successfully",
            400: "Bad Request: Missing or invalid 'text'",
            502: "Metadata extraction error",
        }
    )
    def post(self):
        try:
            data = request.get_json(silent=True) or {}
            text = data.get("text", "")

            if not isinstance(text, str) or not text.strip():
                return {"error": "'text' debe ser un string no vacío"}, 400

            # Solo Ollama. Host se toma de OLLAMA_HOST (o default) dentro del extractor.
            extractor = MetadataExtractor(
                ollama_llm="llama3.1",
                ollama_embed_model="mxbai-embed-large",
            )
            result = extractor.extract_metadata_from_text(text)

            # Si hubo error, devolver 502; si no, 200 con las tres claves
            if "error" in result:
                logger.error(f"Metadata extraction error: {result['error']}")
                return result, 502

            return result, 200

        except Exception as e:
            logger.exception(f"Metadata extraction exception: {e}")
            return {"error": str(e)}, 502