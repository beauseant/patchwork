from flask_restx import Api

from .ns_corpora import api as ns1
from .ns_collections import api as ns2
from .ns_models import api as ns3
from .ns_queries import api as ns4

api = Api(
    title="NP's Solr Service API",
    version='1.0',
    description='This RESTful API utilizes the Solr search engine for efficient data storage and retrieval of logical corpora and their associated topic models. The API also offers a range of query options to facilitate information retrieval.',
)

api.add_namespace(ns2, path='/collections')
api.add_namespace(ns1, path='/corpora')
api.add_namespace(ns3, path='/models')
api.add_namespace(ns4, path='/queries')