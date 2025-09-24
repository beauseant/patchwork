import logging
import os
from dotenv import load_dotenv
from flask import request
from flask_restx import Namespace, Resource, fields
from src.core.metadata_extractor import MetadataExtractor

# -----------------------------
# ✅ Cargar Variables de Entorno
# -----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "La variable de entorno OPENAI_API_KEY no está configurada")

# -----------------------------
# ✅ Configuración del Logger
# -----------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('MetadataExtractor')

# -----------------------------
# ✅ Definición del Namespace en Flask-Restx
# -----------------------------
api = Namespace('Metadata Extraction')

# -----------------------------
# ✅ Definición del Modelo de Entrada para Swagger
# -----------------------------
metadata_model = api.model("MetadataRequest", {
    "text": fields.String(required=True, description="Texto a analizar")
})

# -----------------------------
# ✅ Definición del Endpoint de Extracción de Metadatos
# -----------------------------


@api.route('/extract/')
class ExtractMetadata(Resource):
    @api.expect(metadata_model)  # 💡 Esto añade el input en Swagger
    @api.doc(
        responses={
            200: 'Success: Metadata extracted successfully',
            400: 'Bad Request: Missing required parameters or invalid data',
            502: 'Metadata extraction error',
        }
    )
    def post(self):

        try:
            # 🚀 Obtener datos directamente del JSON en el body
            data = request.get_json()

            # Validar que 'text' exista y sea un string
            if not data or 'text' not in data or not isinstance(data['text'], str):
                return {"error": "'text' must be a non-empty string"}, 400

            text = data['text']

            # Inicializar el extractor de metadatos
            extractor = MetadataExtractor(api_key=api_key, logger=logger)

            # Ejecutar la extracción de metadatos
            result = extractor.extract_metadata_from_text(text)

            # ✅ Corregido: Devuelve el resultado sin usar jsonify()
            return result, 200

        except Exception as e:
            logger.error(f"❌ Metadata extraction error: {str(e)}")
            return {
                "status": 502,
                "error": str(e)
            }, 502
