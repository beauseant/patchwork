"""
This script defines a Flask RESTful namespace for extracting text from uploaded PDF files.

Author: Lorena Calvo-Bartolomé
Date: 03/04/2024
"""

import logging
import time
import os
from flask_restx import Namespace, Resource, reqparse
from werkzeug.datastructures import FileStorage
from src.core.pdf_extractor.processPDFn import PDFTextExtractor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("PDFExtractor")

# ======================================================
# Define namespace for PDF text extraction
# ======================================================
api = Namespace("PDF Extraction")

# ======================================================
# Define parser to accept file uploads
# ======================================================
file_upload_parser = reqparse.RequestParser()
file_upload_parser.add_argument(
    "file", type=FileStorage, location="files", required=True, help="PDF file to extract text from."
)

# ======================================================
# Create PDF Parser object (once)
# ======================================================
#pdf_parser = PDFParser(
#    extract_header_footer=False,
#    generate_img_desc=False,
#    generate_table_desc=False,
#)

pdf_parser = PDFTextExtractor()

@api.route("/extract_text/")
class extract_text(Resource):
    @api.doc(
        parser=file_upload_parser,
        responses={
            200: "Success: Text extracted successfully",
            400: "Bad Request: Invalid input",
            500: "Server Error: Failed to process the PDF",
        },
    )
    def post(self):
        start_time = time.time()
        args = file_upload_parser.parse_args()
        uploaded_file = args["file"]

        if not uploaded_file or not uploaded_file.filename.lower().endswith(".pdf"):
            return {"error": "Invalid file. Please upload a valid PDF."}, 400

        temp_dir = "/tmp"
        temp_pdf_path = os.path.join(temp_dir, uploaded_file.filename)

        try:
            # Save file temporarily
            uploaded_file.save(temp_pdf_path)

            # Extract text
            #raw_text = pdf_parser.extract_raw_text(temp_pdf_path, base_output_dir=temp_dir)
            raw_text = pdf_parser.extraerTexto(temp_pdf_path)

            elapsed = time.time() - start_time
            response = {
                "responseHeader": {"status": 200, "time": elapsed},
                "response": {"text": raw_text},
            }
            logger.info("Text extracted successfully")
            return response, 200

        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}", exc_info=True)
            return {"error": "Failed to process the PDF", "details": str(e)}, 500

        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)
            except Exception:
                logger.warning("Failed to remove temporary file: %s", temp_pdf_path)
