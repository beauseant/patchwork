"""
This script defines a Flask RESTful namespace for extracting text from uploaded PDF files.

Author: Lorena Calvo-Bartolomé
Date: 03/04/2024
"""

import logging
import time
import os
from flask_restx import Namespace, Resource, reqparse
from flask import request
from werkzeug.datastructures import FileStorage
from src.core.pdf_parser import PDFparser

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
# Create PDF Parser object
# ======================================================
pdf_parser = PDFparser(logger=logger)


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

        if not uploaded_file or not uploaded_file.filename.endswith(".pdf"):
            return {"error": "Invalid file. Please upload a valid PDF."}, 400

        try:
            # Save file temporarily
            temp_dir = "/tmp"
            temp_pdf_path = os.path.join(temp_dir, uploaded_file.filename)
            uploaded_file.save(temp_pdf_path)

            # Extract text from PDF
            extracted_text = pdf_parser.parse(temp_pdf_path)

            end_time = time.time() - start_time
            response = {
                "responseHeader": {"status": 200, "time": end_time},
                "response": {"text": extracted_text},
            }
            logger.info("Text extracted successfully")
            
            # Clean up temporary file
            os.remove(temp_pdf_path)
            
            return response, 200

        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return {"error": "Failed to process the PDF", "details": str(e)}, 500
