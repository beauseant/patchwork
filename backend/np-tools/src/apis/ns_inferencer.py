"""
This script defines a Flask RESTful namespace for performing inference.

Author: Lorena Calvo-Bartolomé
Date: 19/03/2024
"""

import os
import pathlib
import time
from flask_restx import Namespace, Resource, reqparse # type: ignore

from src.core.inferencer import Inferencer

import logging
logging.basicConfig(level='DEBUG')
logger = logging.getLogger('Inferencer')

# ======================================================
# Define namespace for inference operations
# ======================================================
api = Namespace('Inference')

# ======================================================
# Create Inferencer object
# ======================================================
inferencer = Inferencer(logger=logger)

# ======================================================
# Define parsers to take inputs from user
# ======================================================
infer_doc_parser = reqparse.RequestParser()
#infer_doc_parser.add_argument('model_for_infer', help='Model to be used for the inference', required=True)
infer_doc_parser.add_argument(
    'text_to_infer',
    help='Input text to be inferred. Separate multiple texts with commas.',
    required=False
)
infer_doc_parser.add_argument(
    'cpv',
    help='The first two digits of the CPV code corresponding to the text.',
    required=False
)
infer_doc_parser.add_argument(
    'granularity',
    help='Specifies the level of topic detail: "large" (more topics) or "small" (fewer topics). Default is "large".',
    default="large",
    required=False
)
infer_doc_parser.add_argument(
    'model_name',
    help="Model to be used for the inference in case 'cpv' and 'granularity' are not provided.",
    required=False
)

@api.route('/predict/')
class predict(Resource):
    """Given a text and a model (trained topic model), this endpoint returns the inference of the text using the model.
    """
    @api.doc(
        parser=infer_doc_parser,
        responses={
            200: 'Success: Inference generated successfully',
            501: 'Model for inference not found: An error occurred while trying to find the model for inference',
            502: 'Inference generation error: An error occurred while generating the inference'
        })
    def post(self):

        start_time = time.time()

        args = infer_doc_parser.parse_args()

        text_to_infer = args['text_to_infer']
        cpv = args["cpv"]
        granularity = args["granularity"]
        model_name = args["model_name"]
        
        if cpv is None and model_name is None:
            # return error
            end_time = time.time() - start_time
            sc = 503
            responseHeader = {
                "status": sc,
                "time": end_time,
                "error": "CPV code or model name not provided."
            }
            response = {
                "responseHeader": responseHeader,
                "response": None
            }
            return response, sc
        elif cpv is not None:
            model_to_infer = f"{cpv}_{granularity}"
        else:
            model_to_infer = model_name
        
        # We look for the model in case the user did not write the name properly
        look_dir = pathlib.Path("/data/source/cpv_models")
        model_path = None
        for folder in os.listdir(look_dir):
            if folder.lower() == model_to_infer.lower():
                model_path = folder
                logger.info(f"-- -- Model found at: {model_path}")

        if model_path is not None:
            model_for_infer = look_dir / model_path
            logger.info(
            f"-- -- Model for inference: {model_for_infer.as_posix()}")
        else:        
            model_for_infer = look_dir / f"default_{granularity}"
            if not model_for_infer.is_dir():
                logger.error(
                    f"-- -- Model for inference not found: { args["model_for_infer"]} and default model is also not available.")
                end_time = time.time() - start_time
                sc = 501
                responseHeader = {
                    "status": sc,
                    "time": end_time,
                    "error": f"Model for inference not found: { args["model_for_infer"]}"
                }
                response = {
                    "responseHeader": responseHeader,
                    "response": None
                }
                return response, sc
            else:
                logger.info(f"Using default model for inference: {model_for_infer}")

        # Perform inference
        if isinstance(text_to_infer, str):
            text_to_infer_lst = text_to_infer.split(",")
            if len(text_to_infer_lst) == 1:
                text_to_infer_lst = [text_to_infer]
        elif isinstance(text_to_infer, list):
            text_to_infer_lst = text_to_infer
        else:
            end_time = time.time() - start_time
            sc = 502
            responseHeader = {
                "status": sc,
                "time": end_time,
                "error": str(e)
            }
            return response, sc
        
        try:
            thetas = inferencer.predict(
                texts=text_to_infer_lst,
                model_for_infer_path=model_for_infer
            )

            end_time = time.time() - start_time

            # Generate response
            sc = 200
            responseHeader = {
                "status": sc,
                "time": end_time
            }

            response = {
                "responseHeader": responseHeader,
                "response": thetas
            }

            return response, sc

        except Exception as e:
            end_time = time.time() - start_time
            sc = 502
            responseHeader = {
                "status": sc,
                "time": end_time,
                "error": str(e)
            }

            response = {
                "responseHeader": responseHeader,
                "response": None
            }

            return response, sc
