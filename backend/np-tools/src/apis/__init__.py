from flask_restx import Api, Resource # type: ignore
from datetime import datetime

#from .ns_embedder import api as ns1
#from .ns_inferencer import api as ns2
#from .ns_lemmas import api as ns3
from .ns_predict_cpv import api as ns4
from .ns_pdf_parser import api as ns5
from .ns_objective_extractor import api as ns6
from .ns_extract_metadata import api as ns7

api = Api(
    title="NP Tools API",
    version="1.0",
    description=(
        "A RESTful API providing auxiliary endpoints for tender processing, including:\n"
        "- Predicting document-topic representation using a trained model.\n"
        "- Extracting lemmas from text.\n"
        "- Predicting the CPV (Common Procurement Vocabulary) code.\n"
        "- Extracting raw text from PDF files.\n"
        "- Extracting objectives from documents.\n"
        "- Extracting metadata from documents."
    )
)

#api.add_namespace(ns1, path='/embedding')
#api.add_namespace(ns2, path='/inference')
#api.add_namespace(ns3, path='/lemmatization')
api.add_namespace(ns4, path='/cpv')
api.add_namespace(ns5, path='/pdf')
api.add_namespace(ns6, path='/objective')
api.add_namespace(ns7, path='/metadata')

@api.route('/ping')
class Ping(Resource):
    def get(self):
        return {
            "status": "pong",
            "timestamp": datetime.now().isoformat() + 'Z',
            "service": "NP Tools API"
        }